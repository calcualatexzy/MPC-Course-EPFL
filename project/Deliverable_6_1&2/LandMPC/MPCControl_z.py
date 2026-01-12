import numpy as np
import cvxpy as cp
from control import dlqr
from mpt4py import Polyhedron

from .MPCControl_base import MPCControl_base


class MPCControl_z(MPCControl_base):
    """
    Pure Tube MPC controller for Z position subsystem using polyhedrons.
    
    States: [vz, z]
    Input: [Pavg]
    Disturbance: w ∈ W = [-15, 5]
    
    Uses Tube MPC with:
    - Ancillary controller K for robustness
    - Minimal robust positively invariant (mRPI) set E (polyhedron)
    - Constraint tightening based on mRPI set
    - Terminal invariant set Xf (polyhedron)
    - Tube MPC control law: u = v + K(x - z)
    """
    x_ids: np.ndarray = np.array([8, 11])  # vz, z
    u_ids: np.ndarray = np.array([2])      # Pavg

    # Tube MPC components
    K: np.ndarray      # Ancillary controller gain
    E: Polyhedron = None  # mRPI set (polyhedron form)
    Xf: Polyhedron = None  # Terminal set (polyhedron form)
    P: np.ndarray      # Terminal cost matrix
    
    # Tightened constraint sets (polyhedrons)
    X_tilde: Polyhedron = None  # Tightened state constraints: X_tilde = X ⊖ E
    U_tilde: Polyhedron = None  # Tightened input constraints: U_tilde = U ⊖ KE
    Xf_tilde: Polyhedron = None  # Tightened terminal set: Xf_tilde = Xf ⊖ E

    # CVXPY variables - Tube MPC uses nominal variables z and v
    z_var: cp.Variable  # Nominal state trajectory z
    v_var: cp.Variable  # Nominal input trajectory v
    x0_param: cp.Parameter  # Actual current state x0
    x_target_param: cp.Parameter
    u_target_param: cp.Parameter
    
    # Disturbance bounds
    w_min: float = -15.0
    w_max: float = 5.0

    def _compute_rpi_polyhedron(self, A_cl: np.ndarray, B: np.ndarray, w_min: float, w_max: float, max_iter: int = 50) -> Polyhedron:
        """
        Compute minimal robust positively invariant (mRPI) set as a polyhedron.
        
        Uses iterative algorithm: E = ⊕_{i=0}^{∞} A_cl^i * B * W
        where ⊕ denotes Minkowski sum and W is the scalar disturbance set [w_min, w_max].
        
        Based on exercise 7 solution.
        
        Args:
            A_cl: Closed-loop system matrix A + B*K
            B: Input matrix that maps scalar disturbance w to state space
            w_min: Minimum disturbance value
            w_max: Maximum disturbance value
            max_iter: Maximum number of iterations
            
        Returns:
            E: mRPI set as a Polyhedron
        """
        nx = A_cl.shape[0]
        
        # Create disturbance set W in state space: W = {B*w : w_min <= w <= w_max}
        # First create scalar disturbance set as 1D polyhedron
        W_scalar = Polyhedron.from_Hrep(
            A=np.array([[1], [-1]]),  # w <= w_max and -w <= -w_min
            b=np.array([w_max, -w_min])
        )
        
        # Apply affine map B to get W in state space: W = B @ W_scalar
        W = B @ W_scalar
        
        # Initialize with W (this is i=0 term)
        Omega = W
        itr = 0
        A_cl_ith_power = np.eye(nx)
        
        while itr < max_iter:
            A_cl_ith_power = np.linalg.matrix_power(A_cl, itr)
            Omega_next = Omega + A_cl_ith_power @ W
            Omega_next.minVrep()
            # print(np.linalg.norm(A_cl_ith_power, ord=2))
            if np.linalg.norm(A_cl_ith_power, ord=2) < 0.02:
                print(f'Minimal robust invariant set computation converged after {itr} iterations.')
                break
            
            if itr == max_iter - 1:
                print(f'Minimal robust invariant set computation did NOT converge after {max_iter} iterations.')
            
            Omega = Omega_next
            itr += 1
        
        return Omega_next

    def _compute_terminal_set_polyhedron(self, A_cl: np.ndarray, X_tilde: Polyhedron, U_tilde: Polyhedron, K: np.ndarray, max_iter: int = 50) -> Polyhedron:
        """
        Compute terminal invariant set Xf as a polyhedron using maximal invariant set algorithm.
        
        The terminal set satisfies: Xf = {x : x in X_tilde, K*x in U_tilde, A_cl*x in Xf}
        
        Based on exercise 3 solution.
        
        Args:
            A_cl: Closed-loop system matrix A + B*K
            X_tilde: Tightened state constraint set (as Polyhedron)
            U_tilde: Tightened input constraint set (as Polyhedron)
            K: Terminal controller gain
            max_iter: Maximum number of iterations
            
        Returns:
            Xf: Terminal invariant set as a Polyhedron
        """
        # Constraint set: X_tilde ∩ {x : K*x in U_tilde}
        # Convert input constraint to state constraint: K*x in U_tilde
        # U_tilde = {u : A_u * u <= b_u}
        # So: {x : A_u * K * x <= b_u}
        KU_tilde = Polyhedron.from_Hrep(A=U_tilde.A @ K, b=U_tilde.b)
        
        # Intersect state and input constraints
        O = X_tilde.intersect(KU_tilde)
        
        # Iterative computation of maximal invariant set
        # Based on exercise 3 solution
        itr = 1
        converged = False
        
        while itr < max_iter:
            O_prev = O
            F, f = O.A, O.b
            
            # Compute the pre-set: O = {x : F*x <= f, F*A_cl*x <= f}
            # This is equivalent to: O = O ∩ {x : A_cl*x in O}
            O = Polyhedron.from_Hrep(
                np.vstack((F, F @ A_cl)), 
                np.vstack((f, f)).reshape((-1,))
            )
            O.minHrep()
            
            if O == O_prev:
                converged = True
                break
            
            print(f'Iteration {itr}... not yet converged')
            itr += 1
        
        if converged:
            print(f'Maximum invariant set successfully computed after {itr} iterations.')
        else:
            print(f'Warning: Terminal set computation did NOT converge after {max_iter} iterations')
        
        return O

    def _setup_controller(self) -> None:
        """
        Setup pure tube MPC with polyhedron-based mRPI set and terminal set.
        """
        Q = np.diag([50.0, 400.0])
        R = np.diag([0.08]) 
        
        K_lqr, P, _ = dlqr(self.A, self.B, Q, R)
        self.K = -K_lqr
        self.P = P
        self.Q = Q
        self.R = R
        
        # Check stability of closed-loop system
        A_cl = self.A + self.B @ self.K
        
        # Compute mRPI set E as polyhedron
        self.E = self._compute_rpi_polyhedron(A_cl, self.B, self.w_min, self.w_max, max_iter=70)
        
        # Create original constraint sets as polyhedra
        # State constraints: z >= 0 (in delta coordinates: z_delta >= -z_s)
        # For vz: no hard constraint, use large bounds
        vz_max = 50.0
        z_min_delta = -self.xs[1]  # z >= 0 => z_delta >= -z_s
        
        # Original X as polyhedron
        X_A = np.array([
            [1, 0],   # vz <= vz_max
            [-1, 0],  # -vz <= vz_max (i.e., vz >= -vz_max)
            [0, -1]   # -z <= -z_min_delta (i.e., z >= z_min_delta)
        ])
        X_b = np.array([vz_max, vz_max, -z_min_delta])
        X = Polyhedron.from_Hrep(A=X_A, b=X_b)
        
        # Compute X_tilde = X ⊖ E (Pontryagin difference)
        self.X_tilde = X - self.E
        self.X_tilde.minHrep()
        
        # Create original U as polyhedron (tightened input constraints)
        # Original input constraints in delta coordinates
        u_min_delta = np.array([40.0]) - self.us
        u_max_delta = np.array([80.0]) - self.us
        U_A = np.vstack([np.eye(self.nu), -np.eye(self.nu)])
        U_b = np.hstack([u_max_delta, -u_min_delta])
        U = Polyhedron.from_Hrep(A=U_A, b=U_b)
        
        # Compute KE = {K*e : e in E}
        KE = self.K @ self.E
        self.U_tilde = U - KE
        self.U_tilde.minHrep()
        self.Xf = self._compute_terminal_set_polyhedron(A_cl, self.X_tilde, self.U_tilde, self.K, max_iter=50)
        
        # Setup CVXPY optimization - Tube MPC uses nominal variables z and v
        self.z_var = cp.Variable((self.nx, self.N + 1))  # Nominal state z
        self.v_var = cp.Variable((self.nu, self.N))      # Nominal input v
        
        self.x0_param = cp.Parameter(self.nx)  # Actual current state x0
        self.x_target_param = cp.Parameter(self.nx)
        self.u_target_param = cp.Parameter(self.nu)
        
        self.x_target_param.value = np.zeros(self.nx)
        self.u_target_param.value = np.zeros(self.nu)
        
        # Build cost function - simple quadratic cost
        cost = 0
        
        # Stage cost - cost on nominal variables z and v
        for k in range(self.N):
            dz = self.z_var[:, k] - self.x_target_param  # Nominal state deviation
            dv = self.v_var[:, k] - self.u_target_param  # Nominal input deviation
            
            # Tracking cost on nominal trajectory
            cost += cp.quad_form(dz, self.Q) + cp.quad_form(dv, self.R)
        
        # Terminal cost on nominal state
        dz_N = self.z_var[:, self.N] - self.x_target_param
        cost += cp.quad_form(dz_N, self.P)
        
        # Build constraints - Pure Tube MPC structure
        constraints = []
        
        # Initial condition: x0 in z0 + E
        # This means: E.A @ (x0 - z0) <= E.b
        constraints.append(self.E.A @ (self.x0_param - self.z_var[:, 0]) <= self.E.b)
        
        # Dynamics on nominal variables z
        for k in range(self.N):
            constraints.append(
                self.z_var[:, k+1] == self.A @ self.z_var[:, k] + self.B @ self.v_var[:, k]
            )
            
            # X_tilde constraints: z_k in X_tilde
            constraints.append(self.X_tilde.A @ self.z_var[:, k] <= self.X_tilde.b)
            
            # U_tilde constraints: v_k in U_tilde
            constraints.append(self.U_tilde.A @ self.v_var[:, k] <= self.U_tilde.b)
        
        # Terminal constraint: z_N in Xf
        # constraints.append(self.Xf.A @ self.z_var[:, self.N] <= self.Xf.b)
        
        self.ocp = cp.Problem(cp.Minimize(cost), constraints)
        print("Pure Tube MPC setup complete")

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve tube MPC and return control with ancillary feedback.
        
        The actual control is: u = v + K(x - z)
        where v is the nominal control, z is the nominal state.
        
        Args:
            x0: Current state in ABSOLUTE coordinates
            x_target: Target state (optional)
            u_target: Target input (optional)
            
        Returns:
            u0: Optimal control input in ABSOLUTE coordinates
            z_traj: Nominal state trajectory
            v_traj: Nominal control trajectory
        """
        if self.E is None or self.Xf is None:
            # Fallback: use ancillary controller only
            x0_delta = x0 - self.xs
            u0_delta = self.K @ x0_delta
            u0 = np.clip(u0_delta + self.us, 40.0, 80.0)
            return u0, np.zeros((self.nx, self.N + 1)), np.zeros((self.nu, self.N))
        
        # Convert initial state from ABSOLUTE to DELTA coordinates
        x0_delta = x0 - self.xs
        
        # Set parameters
        self.x0_param.value = x0_delta
        
        if x_target is not None:
            self.x_target_param.value = x_target - self.xs
        else:
            self.x_target_param.value = np.zeros(self.nx)
        
        if u_target is not None:
            self.u_target_param.value = u_target - self.us
        else:
            self.u_target_param.value = np.zeros(self.nu)
        
        # Solve Tube MPC
        try:
            self.ocp.solve(
                solver=cp.OSQP,
                warm_start=True,
                verbose=False,
                max_iter=5000,
                eps_abs=1e-3,
                eps_rel=1e-3,
                polish=True,
                adaptive_rho=True
            )
        except Exception as e:
            # Fallback: use ancillary controller
            print(f"Warning: Solver failed: {e}")
            u0_delta = self.K @ x0_delta
            u0 = np.clip(u0_delta + self.us, 40.0, 80.0)
            return u0, np.zeros((self.nx, self.N + 1)), np.zeros((self.nu, self.N))
        
        # Handle solver status
        if self.ocp.status not in ["optimal", "optimal_inaccurate"]:
            # Fallback: use ancillary controller
            print(f"Warning: Solver status: {self.ocp.status}")
            u0_delta = self.K @ x0_delta
            u0 = np.clip(u0_delta + self.us, 40.0, 80.0)
            return u0, np.zeros((self.nx, self.N + 1)), np.zeros((self.nu, self.N))
        
        # Extract nominal solution: z (nominal state) and v (nominal input)
        v0 = self.v_var[:, 0].value  # Nominal input v
        z_traj = self.z_var.value  # Nominal state trajectory z
        v_traj = self.v_var.value  # Nominal input trajectory v
        
        if v0 is None or z_traj is None:
            # Fallback: use ancillary controller
            u0_delta = self.K @ x0_delta
            u0 = np.clip(u0_delta + self.us, 40.0, 80.0)
            return u0, np.zeros((self.nx, self.N + 1)), np.zeros((self.nu, self.N))
        
        # Apply Tube MPC control law: u = v + K(x - z)
        # where v is nominal input, z is nominal state, x is actual state
        error = x0_delta - z_traj[:, 0]  # Deviation from nominal
        u0_delta = v0 + self.K @ error  # Actual control in delta coordinates
        
        # Convert to absolute coordinates and clip to physical bounds
        u0 = u0_delta + self.us
        u0 = np.clip(u0, 40.0, 80.0)
        
        if z_traj is None:
            z_traj = np.zeros((self.nx, self.N + 1))
        if v_traj is None:
            v_traj = np.zeros((self.nu, self.N))
        
        return u0, z_traj, v_traj

    # Estimator methods (not used in Part 6)
    def setup_estimator(self):
        self.d_estimate = np.zeros(self.nx)
        self.d_gain = 0.0

    def update_estimator(self, x_data: np.ndarray, u_data: np.ndarray) -> None:
        pass
