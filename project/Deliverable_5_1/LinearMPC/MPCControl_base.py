import cvxpy as cp
import numpy as np
from control import dlqr
from mpt4py import Polyhedron
from scipy.signal import cont2discrete

def max_invariant_set(A_cl, X: Polyhedron, max_iter = 30) -> Polyhedron:
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
	
	# if converged:
	# 	print('Maximum invariant set successfully computed after {0} iterations.'.format(itr))
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

        self._setup_parameters()
        self._setup_controller()

    def _setup_parameters(self) -> None:

        self.Q = 0.1 * np.eye(self.nx)
        self.R = 0.1 * np.eye(self.nu)

        self.params = {}
        # Input constraints
        safety_margin = 0.01
        self.params['delta'] = (-np.deg2rad(15)+safety_margin, np.deg2rad(15)-safety_margin)
        self.params['P_avg'] = (40+safety_margin, 80-safety_margin) 
        self.params['P_diff'] = (-20+safety_margin, 20-safety_margin)
        
        # State constraints
        self.params['alpha'] = (-np.deg2rad(15)+safety_margin, np.deg2rad(15)-safety_margin)
        self.params['beta'] = (-np.deg2rad(15)+safety_margin, np.deg2rad(15)-safety_margin)
        

    def _setup_controller(self) -> None:
        #################################################
        # YOUR CODE HERE

        K, Qf, _ = dlqr(self.A, self.B, self.Q, self.R)
        self.K = -K
        self.Qf = Qf
        self.A_cl = self.A + self.B @ self.K

        KU = Polyhedron.from_Hrep(self.U.A @ self.K, self.U.b)

        self.O_inf = max_invariant_set(self.A_cl, self.X.intersect(KU))

        self.x_var = cp.Variable((self.nx, self.N + 1), name='x')
        self.u_var = cp.Variable((self.nu, self.N), name='u')
        self.x0_var = cp.Parameter((self.nx,), name='x0')

        # Costs 
        self.cost = 0

        for i in range(self.N):
            self.cost += cp.quad_form(self.x_var[:, i], self.Q)
            self.cost += cp.quad_form(self.u_var[:, i], self.R)
        self.cost += cp.quad_form(self.x_var[:, -1], self.Qf)

        # YOUR CODE HERE
        #################################################

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

        if x_target is None:
            x_target = self.xs
        if u_target is None:
            u_target = self.us

        self.x0_var.value = x0 - x_target

        # Constraints
        self.constraints = []
        # Initial condition
        self.constraints.append(self.x_var[:, 0] == self.x0_var)
        # System dynamics
        self.constraints.append(self.x_var[:, 1:] == self.A @ self.x_var[:, :-1] + self.B @ self.u_var)
        # State constraints
        # self.constraints.append(self.X.A @ self.x_var[:, :-1] <= self.X.b.reshape(-1, 1))
        # Input constraints
        # self.constraints.append(self.U.A @ self.u_var <= self.U.b.reshape(-1, 1))
        # Terminal Constraints

        #self.constraints.append(self.O_inf.A @ self.x_var[:, -1] <= self.O_inf.b.reshape(-1, 1))

        # State constraints for delta form
        self.constraints.append(self.X.A @ self.x_var[:, :-1] <= (self.X.b - self.X.A @ x_target).reshape(-1, 1))

        # Input constraints for delta form
        self.constraints.append(self.U.A @ self.u_var <= (self.U.b - self.U.A @ u_target).reshape(-1, 1))

        # Terminal Constraints (slightly larger than the original form)
        # KU = Polyhedron.from_Hrep(self.U.A @ self.K, self.U.b - self.U.A @ u_target)

        # X = Polyhedron.from_Hrep(self.X.A, self.X.b - self.X.A @ x_target)

        # self.O_inf = max_invariant_set(self.A_cl, X.intersect(KU))
        # self.constraints.append(self.O_inf.A @ self.x_var[:, -1] <= self.O_inf.b.reshape(-1, 1))

        self.ocp = cp.Problem(cp.Minimize(self.cost), self.constraints)

        self.ocp.solve()

        u0 = self.u_var.value[:, 0] + u_target
        x_traj = self.x_var.value[:, :] + x_target.reshape(-1, 1)
        u_traj = self.u_var.value[:, :] + u_target.reshape(-1, 1)

        # YOUR CODE HERE
        #################################################

        return u0, x_traj, u_traj
