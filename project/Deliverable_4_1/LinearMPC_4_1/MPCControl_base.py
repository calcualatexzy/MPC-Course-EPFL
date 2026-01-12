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
    ny: int
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
        
        self._setup_controller()
        self._setup_steady_state_controller()

    def _setup_steady_state_controller(self) -> None:

        self.xs_var = cp.Variable(self.nx)
        self.us_var = cp.Variable(self.nu)
        self.y_target_param = cp.Parameter(self.ny)

        # Default reference
        self.y_target_param.value = np.zeros(self.ny)

        # cost
        cost = cp.quad_form(self.us_var, np.eye(self.nu))

        # constraints
        constraints = []

        # x = Ax + Bu
        constraints.append(self.xs_var == self.A @ self.xs_var + self.B @ self.us_var)

        # y = Cx
        constraints.append(self.C @ self.xs_var == self.y_target_param)

        # input constraints
        constraints.append(self.U.A @ self.us_var <= self.U.b)
        constraints.append(self.X.A @ self.xs_var <= self.X.b)

        self.steady_state_ocp = cp.Problem(cp.Minimize(cost), constraints)

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

        self.X_f = self._compute_terminal_set(self.X, self.U)

        self.x_var = cp.Variable((self.nx, self.N + 1), name='x')
        self.u_var = cp.Variable((self.nu, self.N), name='u')
        self.x0_param = cp.Parameter(self.nx)
        self.x_target_param = cp.Parameter(self.nx)
        self.u_target_param = cp.Parameter(self.nu)

        # Default references (zero for stabilization)
        self.x_target_param.value = np.zeros(self.nx)
        self.u_target_param.value = np.zeros(self.nu)

        # Slack variables for soft state constraints
        n_state_constraints = self.X.A.shape[0]
        self.s_var = cp.Variable((n_state_constraints, self.N), name='s')
        
        # Penalty weight for slack variables (high penalty to discourage constraint violation)
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
            # Dynamics
            constraints.append(
                self.x_var[:, k + 1] == self.A @ self.x_var[:, k] + self.B @ self.u_var[:, k]
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

    def _C(self) -> np.ndarray:
        """
        Get the measure matrix C.
        """
        raise NotImplementedError("subclass must implement _C()")

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

        # if x_target is not None:
        #     self.y_target_param.value = self.C @ (x_target - self.xs)

        #     self.steady_state_ocp.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        #     self.x_target_param.value = self.xs_var.value
        #     self.u_target_param.value = self.us_var.value
        if x_target is not None:
            self.x_target_param.value = x_target - self.xs
        if u_target is not None:
            self.u_target_param.value = u_target - self.us

        # Solve MPC problem
        self.ocp.solve(solver=cp.OSQP, warm_start=True, verbose=False)

        u0 = self.u_var[:, 0].value + self.us.reshape(-1, 1)
        x_traj = self.x_var.value + self.xs.reshape(-1, 1)
        u_traj = self.u_var.value + self.us.reshape(-1, 1)
        # YOUR CODE HERE
        #################################################

        return u0, x_traj, u_traj
