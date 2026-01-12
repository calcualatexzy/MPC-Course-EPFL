import numpy as np

from LinearMPC_5_1.MPCControl_base import MPCControl_base
from mpt4py import Polyhedron
import cvxpy as cp
from control import dlqr

from typing import Tuple
class MPCControl_zvel(MPCControl_base):
    x_ids: np.ndarray = np.array([8])
    u_ids: np.ndarray = np.array([2])

    # only useful for part 5 of the project
    d_estimate: np.ndarray
    d_gain: float

    """
    x: [v_z]
    u: P_avg
    """
    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        xs: np.ndarray,
        us: np.ndarray,
        Ts: float,
        H: float,
    ) -> None:
    
        self.Ts = Ts
        self.H = H
        self.N = int(H / Ts)
        self.nx = self.x_ids.shape[0]
        self.nu = self.u_ids.shape[0]
        self.C = self._C()
        self.ny = self.C.shape[0]

        # System definition
        xids_xi, xids_xj = np.meshgrid(self.x_ids, self.x_ids)
        A_red = A[xids_xi, xids_xj].T
        uids_xi, uids_xj = np.meshgrid(self.x_ids, self.u_ids)
        B_red = B[uids_xi, uids_xj].T
        self.A, self.B = self._discretize(A_red, B_red, Ts)
        self.xs = xs[self.x_ids]
        self.us = us[self.u_ids]

        self.params = {}
        # Input constraints
        self.params['delta'] = (-np.deg2rad(15)+0.05, np.deg2rad(15)-0.05)
        self.params['P_avg'] = (40+2, 80-2) 
        self.params['P_diff'] = (-20, 20)
        
        # State constraints
        self.params['alpha'] = (-np.deg2rad(10), np.deg2rad(10))
        self.params['beta'] = (-np.deg2rad(10), np.deg2rad(10))
        
        self.setup_estimator()
        # Create shared parameter before setting up controllers
        self.d_estimate_param = cp.Parameter(self.nu, name='d_estimate')
        self.d_estimate_param.value = np.zeros(self.nu)
        
        self._setup_controller_with_disturbance()
        self._setup_steady_state_controller()

        
    def _setup_steady_state_controller(self) -> None:

        self.xs_var = cp.Variable(self.nx)
        self.us_var = cp.Variable(self.nu)
        slack = cp.Variable(self.nx)
        self.y_target_param = cp.Parameter(self.ny, name='y_target')

        # Default reference
        self.y_target_param.value = np.zeros(self.ny)

        # cost
        cost = cp.quad_form(self.us_var, np.eye(self.nu))
        cost += 1e8 * cp.sum_squares(slack)
        # constraints
        constraints = []

        # Steady-state equation in deviation form: xs = A*xs + B*us + B*d
        # This means the system is at equilibrium: (I-A)*xs = B*(us + d)
        constraints.append(self.xs_var == self.A @ self.xs_var + self.B @ self.us_var + self.B @ self.d_estimate_param + slack)

        # y = Cx (output equation)
        constraints.append(self.C @ self.xs_var == self.y_target_param)

        # input constraints
        constraints.append(self.U.A @ self.us_var <= self.U.b)
        # constraints.append(self.X.A @ self.xs_var <= self.X.b)

        self.steady_state_ocp = cp.Problem(cp.Minimize(cost), constraints)


    def _cost_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get cost matrices Q and R.
        """
        Q = np.diag([10])
        R = np.diag([0.005])
        return Q, R

    def _constraints(self) -> tuple[Polyhedron, Polyhedron]:
        """
        Get constraint bounds (X, U).
        """
        # Input constraints: u in U = { u | Mu <= m }
        # P_avg: 40 <= P_avg <= 80
        M = np.array([[1], [-1]])  # [P_avg <= P_avg_max, -P_avg <= -P_avg_min]
        m = np.array([self.params['P_avg'][1] - self.us[0], -self.params['P_avg'][0] + self.us[0]])
        U = Polyhedron.from_Hrep(M, m)
        # State constraints: x in X = { x | Fx <= f }
        # v_z: -np.inf <= v_z <= np.inf
        F = np.array([
            [0],   # v_z <= v_z_max
        ])
        f = np.array([1])
        X = Polyhedron.from_Hrep(F, f)
        return X, U

    def _C(self) -> np.ndarray:
        """
        Get the measure matrix C.
        """
        return np.array([[1.0]])
    
    def _setup_controller_with_disturbance(self) -> None:
        """
        Setup controller with disturbance compensation.
        Overrides base class to include disturbance term in dynamics.
        """
        Q, R = self._cost_matrices()
        self.Q = Q
        self.R = R
        self.X, self.U = self._constraints()

        K, P, _ = dlqr(self.A, self.B, Q, R)
        self.K = -K
        self.P = P

        self.X_f = self._compute_terminal_set(self.X, self.U)

        self.x_var = cp.Variable((self.nx, self.N + 1), name='x')
        self.u_var = cp.Variable((self.nu, self.N), name='u')
        self.x0_param = cp.Parameter(self.nx, name='x0')
        self.x_target_param = cp.Parameter(self.nx, name='x_target')
        self.u_target_param = cp.Parameter(self.nu, name='u_target')
        # Reuse d_estimate_param created in __init__

        # Default references (zero for stabilization)
        self.x_target_param.value = np.zeros(self.nx)
        self.u_target_param.value = np.zeros(self.nu)

        # Slack variables for soft state constraints
        n_state_constraints = self.X.A.shape[0]
        self.s_var = cp.Variable((n_state_constraints, self.N), name='s')
        
        # Penalty weight for slack variables
        rho = 1000.0

        # Costs 
        cost = 0

        for i in range(self.N):
            dx = self.x_var[:, i] - self.x_target_param
            du = self.u_var[:, i] - self.u_target_param
            cost += cp.quad_form(dx, self.Q) + cp.quad_form(du, self.R)
            
            # Penalty for slack variables (L2 + L1 penalty)
            cost += rho * (cp.sum_squares(self.s_var[:, i]) + cp.sum(self.s_var[:, i]))
            
        dx_last = self.x_var[:, -1] - self.x_target_param
        cost += cp.quad_form(dx_last, self.P)

        # Constraints
        constraints = []
        # Initial condition
        constraints.append(self.x_var[:, 0] == self.x0_param)
        for k in range(self.N):
            # Dynamics with disturbance: x^+ = Ax + Bu + Bd*d
            # CRITICAL: Must include disturbance term for offset-free tracking!
            # Where Bd = B (disturbance enters through same channel as input)
            constraints.append(
                self.x_var[:, k + 1] == self.A @ self.x_var[:, k] + self.B @ self.u_var[:, k] + self.B @ self.d_estimate_param
            )

            # Soft state constraints (with slack variables)
            constraints.append(self.X.A @ self.x_var[:, k] <= self.X.b + self.s_var[:, k])

            # Input constraints (keep hard)
            constraints.append(self.U.A @ self.u_var[:, k] <= self.U.b)
            
            # Slack variables must be non-negative
            constraints.append(self.s_var[:, k] >= 0)

        # Terminal Constraints
        dx_last = self.x_var[:, -1] - self.x_target_param
        # constraints.append(self.X_f.A @ dx_last <= self.X_f.b)

        self.ocp = cp.Problem(cp.Minimize(cost), constraints)

    def setup_estimator(self):
        # FOR PART 5 OF THE PROJECT
        ##################################################
        # YOUR CODE HERE
        # For system x^+ = Ax + Bu + Bd, where Bd = B (disturbance enters through input channel)
        # Augmented system: z = [x; d], where d^+ = d (constant disturbance)
        Bd = self.B  # Disturbance matrix (same as input matrix)
        
        # A_hat = [A  Bd;  0  I]
        self.A_hat = np.vstack((
            np.hstack((self.A, Bd)),
            np.hstack((np.zeros((self.nu, self.nx)), np.eye(self.nu)))
        ))
        
        # B_hat = [B; 0]
        self.B_hat = np.vstack((
            self.B,
            np.zeros((self.nu, self.nu))
        ))
        
        # C_hat = [C  Cd], where Cd = 0 (disturbance doesn't appear directly in output)
        Cd = np.zeros((self.ny, self.nu))
        self.C_hat = np.hstack((self.C, Cd))
        
        # Pole placement for observer (moderate eigenvalues as in reference)
        n_poles = self.nx + self.nu
        poles = np.array([0.5, 0.6])
        
        from scipy.signal import place_poles
        res = place_poles(self.A_hat.T, self.C_hat.T, poles)
        self.L = -res.gain_matrix.T
        
        # Initialize estimates (as column vectors)
        self.u_prev = np.zeros((self.nu, 1))
        self.x_hat = np.zeros((self.nx, 1))
        self.d_estimate = np.zeros((self.nu, 1))
        # YOUR CODE HERE
        ##################################################

    def update_estimator(self, y) -> None:
        # FOR PART 5 OF THE PROJECT
        # Following exercise5_offset_free/ex5_sol_offset_free.ipynb
        ##################################################
        # YOUR CODE HERE
        # Current estimate: z_hat = [x_hat; d_estimate]
        z_hat_current = np.vstack((self.x_hat, self.d_estimate))
        
        # Observer update: z_hat_next = A_hat @ z_hat + B_hat @ u + L @ (C_hat @ z_hat - y)
        # CRITICAL: Use CURRENT estimate z_hat_current (not predicted) to compute output error
        # This matches the reference implementation exactly
        z_hat_next = self.A_hat @ z_hat_current + self.B_hat @ self.u_prev + self.L @ (self.C_hat @ z_hat_current - y)

        # Update estimates
        self.x_hat = z_hat_next[:self.nx]
        self.d_estimate = z_hat_next[self.nx:]
        # YOUR CODE HERE
        ##################################################

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        #################################################
        # YOUR CODE HERE
        self.x_target_param.value = np.zeros(self.nx)
        self.u_target_param.value = np.zeros(self.nu)

        # Get measurement (in deviation form): y = C @ (x0 - xs)
        y = self.C @ (x0 - self.xs)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Initialize estimates if this is the first call (should not happen if setup_estimator was called)
        if not hasattr(self, "x_hat") or self.x_hat is None:
            self.x_hat = (x0 - self.xs).reshape(-1, 1)
        if not hasattr(self, "d_estimate") or self.d_estimate is None:
            self.d_estimate = np.zeros((self.nu, 1))
        
        # Ensure estimates are column vectors
        if self.x_hat.ndim == 1:
            self.x_hat = self.x_hat.reshape(-1, 1)
        if self.d_estimate.ndim == 1:
            self.d_estimate = self.d_estimate.reshape(-1, 1)
        
        # Update disturbance estimator
        self.update_estimator(y)
        
        # Update disturbance parameter for MPC and steady-state problems
        self.d_estimate_param.value = self.d_estimate.flatten()

        # Update steady-state target if reference provided
        if x_target is not None:
            self.y_target_param.value = self.C @ (x_target - self.xs).flatten()

            self.steady_state_ocp.solve(solver=cp.OSQP, warm_start=True, verbose=False)
            self.x_target_param.value = self.xs_var.value
            self.u_target_param.value = self.us_var.value

        self.x0_param.value = self.x_hat.flatten()

        # Solve MPC problem
        self.ocp.solve(solver=cp.OSQP, warm_start=True, verbose=False)

        # Extract control input (convert from deviation form back to absolute)
        u0 = (self.u_var[:, 0].value + self.us).flatten()
        x_traj = self.x_var.value + self.xs.reshape(-1, 1)
        u_traj = self.u_var.value + self.us.reshape(-1, 1)

        # Store current input for next estimator update (as column vector)
        # CRITICAL: This must be the input that was actually applied, in deviation form
        self.u_prev = (u0 - self.us).reshape(-1, 1)
        # YOUR CODE HERE
        #################################################

        return u0, x_traj, u_traj