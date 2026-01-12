import cvxpy as cp
import numpy as np
from control import dlqr
from mpt4py import Polyhedron
from scipy.signal import cont2discrete
from mpt4py.base import HData

def max_invariant_set(A_cl, X: Polyhedron, max_iter = 50) -> Polyhedron:
	"""
	Compute invariant set for an autonomous linear time invariant system x^+ = A_cl x
	"""
	O = X
	itr = 1
	converged = False
	# print('Computing maximum invariant set ...')
	while itr < max_iter:
		Oprev = O
		F, f = O.A, O.b
		# Compute the pre-set
		O = Polyhedron.from_Hrep(np.vstack((F, F @ A_cl)), np.vstack((f, f)).reshape((-1,)))
		O.minHrep(True)
		if O == Oprev:
			converged = True
			break
		# print('Iteration {0}... not yet converged'.format(itr))
		itr += 1
	if converged:
		print('Maximum invariant set successfully computed after {0} iterations.'.format(itr))
	return O

class MPCControl_base:
    """Complete states indices"""

    x_ids: np.ndarray
    u_ids: np.ndarray


    """Optimization system"""
    A: np.ndarray
    B: np.ndarray
    xs: np.ndarray
    us: np.ndarray
    nx: int
    nu: int
    Ts: float
    H: float
    N: int

    """Optimization problem"""
    ocp: cp.Problem
    Q: np.ndarray
    R: np.ndarray
    P: np.ndarray # terminal cost
    K: np.ndarray
    X_f: Polyhedron # terminal set
    x_var: cp.Variable
    u_var: cp.Variable
    x0_param: cp.Parameter
    x_target_param: cp.Parameter
    u_target_param: cp.Parameter

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
        safety_margin = 0.0
        self.params['delta'] = (-np.deg2rad(15)+safety_margin, np.deg2rad(15)-safety_margin)
        self.params['P_avg'] = (40+safety_margin, 80+safety_margin) 
        self.params['P_diff'] = (-20+safety_margin, 20+safety_margin)
        
        # State constraints
        self.params['alpha'] = (-np.deg2rad(10)+safety_margin, np.deg2rad(10)-safety_margin)
        self.params['beta'] = (-np.deg2rad(10)+safety_margin, np.deg2rad(10)-safety_margin)
        
        self._setup_controller()

    def _setup_controller(self) -> None:
        #################################################
        # YOUR CODE HERE
        Q, R = self._cost_matrices()
        self.Q = Q
        self.R = R
        self.X, self.U = self._constraints()

        K, P, _ = dlqr(self.A, self.B, Q, R)
        self.K = -K
        self.P = P
        self.A_cl = self.A + self.B @ self.K

        self.X_f = self._compute_terminal_set(self.X, self.U)

        self.x_var = cp.Variable((self.nx, self.N + 1), name='x')
        self.u_var = cp.Variable((self.nu, self.N), name='u')
        self.x0_param = cp.Parameter(self.nx)
        self.x_target_param = cp.Parameter(self.nx)
        self.u_target_param = cp.Parameter(self.nu)

        # Default references (zero for stabilization)
        self.x_target_param.value = np.zeros(self.nx)
        self.u_target_param.value = np.zeros(self.nu)

        # Costs 
        cost = 0

        for i in range(self.N):
            dx = self.x_var[:, i] - self.x_target_param
            du = self.u_var[:, i] - self.u_target_param
            cost += cp.quad_form(dx, self.Q) + cp.quad_form(du, self.R)
        dx_last = self.x_var[:, -1] - self.x_target_param
        cost += cp.quad_form(dx_last, self.P)

        # Constraints
        constraints = []
        # Initial condition
        constraints.append(self.x_var[:, 0] == self.x0_param)
        for k in range(self.N):
            # Dynamics
            constraints.append(
                self.x_var[:, k + 1] == self.A @ self.x_var[:, k] + self.B @ self.u_var[:, k]
            )

            # State constraints
            constraints.append(self.X.A @ self.x_var[:, k] <= self.X.b)

            # Input constraints
            constraints.append(self.U.A @ self.u_var[:, k] <= self.U.b)

        # Terminal Constraints
        dx_last = self.x_var[:, -1] - self.x_target_param
        constraints.append(self.X_f.A @ dx_last <= self.X_f.b)

        self.ocp = cp.Problem(cp.Minimize(cost), constraints)
        
        KU = Polyhedron.from_Hrep(self.U.A @ self.K, self.U.b)
        self.O_inf = max_invariant_set(self.A_cl, self.X.intersect(KU))
        # YOUR CODE HERE
        #################################################

    def _cost_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get cost matrices Q and R.
        """
        raise NotImplementedError("subclass must implement _cost_matrices()")

    def _constraints(self) -> tuple[Polyhedron, Polyhedron]:
        """
        Get constraint bounds (X, U).
        """
        raise NotImplementedError("subclass must implement _constraints()")

    def _compute_terminal_set(self, X: Polyhedron, U: Polyhedron, max_iter = 50) -> Polyhedron:
        """
        Compute maximal positively invariant set for terminal constraint.
        Uses the closed-loop system A_cl = A + B*K and max_invariant_set function.
        """
        # Closed-loop dynamics
        A_cl = self.A + self.B @ self.K

        # Create KU: input constraints under feedback u = K * x
        # U.A @ (K @ x) <= U.b => (U.A @ K) @ x <= U.b
        KU = Polyhedron.from_Hrep(U.A @ self.K, U.b)

        # Intersect state constraints X with input constraints KU
        XU = X.intersect(KU)

        # Compute maximal invariant set using max_invariant_set function
        X_f = max_invariant_set(A_cl, XU, max_iter)

        return X_f

    @staticmethod
    def _discretize(A: np.ndarray, B: np.ndarray, Ts: float):
        nx, nu = B.shape
        C = np.zeros((1, nx))
        D = np.zeros((1, nu))
        A_discrete, B_discrete, _, _, _ = cont2discrete(system=(A, B, C, D), dt=Ts)
        return A_discrete, B_discrete

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        #################################################
        # YOUR CODE HERE
        self.x0_param.value = x0 - self.xs
        self.x_target_param.value = np.zeros(self.nx)
        self.u_target_param.value = np.zeros(self.nu)

        if x_target is not None:
            self.x_target_param.value = x_target - self.xs
        if u_target is not None:
            self.u_target_param.value = u_target - self.us

        # Solve MPC problem
        # import IPython; IPython.embed()
        try:
            self.ocp.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        except Exception as e:
            print(f"MPC solve failed: {e}")
            # Return zero control on failure
            return (
                np.zeros(self.nu),
                np.zeros((self.nx, self.N + 1)),
                np.zeros((self.nu, self.N)),
            )

        u0 = self.u_var[:, 0].value + self.us.reshape(-1, 1)
        x_traj = self.x_var.value + self.xs.reshape(-1, 1)
        u_traj = self.u_var.value + self.us.reshape(-1, 1)
        # YOUR CODE HERE
        #################################################

        return u0, x_traj, u_traj
