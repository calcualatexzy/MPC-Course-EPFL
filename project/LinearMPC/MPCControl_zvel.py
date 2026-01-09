import numpy as np

from .MPCControl_base import MPCControl_base, max_invariant_set
from mpt4py import Polyhedron
import cvxpy as cp


class MPCControl_zvel(MPCControl_base):
    x_ids: np.ndarray = np.array([8])
    u_ids: np.ndarray = np.array([2])

    # only useful for part 5 of the project
    d_estimate: np.ndarray
    d_gain: float

    def _setup_controller(self) -> None:
        """
        x: [v_z]
        u: P_avg
        """
        #################################################
        # YOUR CODE HERE

        print("setting up zvel")
        self.Q = np.diag([10])
        self.R = np.diag([0.5])

        # Input constraints: u in U = { u | Mu <= m }
        # P_avg: 40 <= P_avg <= 80
        self.M = np.array([[1], [-1]])  # [P_avg <= P_avg_max, -P_avg <= -P_avg_min]
        self.m = np.array([self.params['P_avg'][1], -self.params['P_avg'][0]])
        self.U = Polyhedron.from_Hrep(self.M, self.m)
        
        # State constraints: x in X = { x | Fx <= f }
        self.F = np.array([
            [0],   # v_z <= v_z_max
        ])
        self.f = np.array([1])
        self.X = Polyhedron.from_Hrep(self.F, self.f)
    
        super()._setup_controller()

        # YOUR CODE HERE
        #################################################

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

        # State constraints for delta form (soft or hard)
        if self.use_soft_constraints:
            # Soft constraints: allow violation with penalty
            self.constraints.append(
                self.X.A @ self.x_var[:, :-1] <= (self.X.b - self.X.A @ x_target).reshape(-1, 1) + self.epsilon
            )
            total_cost = self.cost + self.slack_penalty_weight * cp.sum(self.epsilon)
        else:
            # Hard constraints
            self.constraints.append(
                self.X.A @ self.x_var[:, :-1] <= (self.X.b - self.X.A @ x_target).reshape(-1, 1)
            )
            total_cost = self.cost

        # Input constraints for delta form
        self.constraints.append(self.U.A @ self.u_var <= (self.U.b - self.U.A @ u_target).reshape(-1, 1))

        # Terminal Constraints (slightly larger than the original form)
        KU = Polyhedron.from_Hrep(self.U.A @ self.K, self.U.b - self.U.A @ u_target)

        X = Polyhedron.from_Hrep(self.X.A, self.X.b - self.X.A @ x_target)

        self.O_inf = max_invariant_set(self.A_cl, X.intersect(KU))
        self.constraints.append(self.O_inf.A @ self.x_var[:, -1] <= self.O_inf.b.reshape(-1, 1))

        self.ocp = cp.Problem(cp.Minimize(total_cost), self.constraints)

        self.ocp.solve()
        # import IPython; IPython.embed();
        u0 = self.u_var.value[:, 0] + u_target
        x_traj = self.x_var.value[:, :] + x_target.reshape(-1, 1)
        u_traj = self.u_var.value[:, :] + u_target.reshape(-1, 1)

        # YOUR CODE HERE
        #################################################

        return u0, x_traj, u_traj

    def setup_estimator(self):
        # FOR PART 5 OF THE PROJECT
        ##################################################
        # YOUR CODE HERE
        # x = A x + B u + Bd d (x = [v_z], Bd = B)
        # y = C x + Cd d (y = v_z)
        self.ny = 1
        self.nd = 1

        self.d_estimate = np.zeros((self.nd, 1))
        self.d_gain = 1.0

        poles = np.array([0.5, 0.6])

        C = np.ones((self.ny, self.nx))
        Bd = self.B

        # A_hat = [A  Bd;  0  I]
        self.A_hat = np.vstack((
            np.hstack((self.A, Bd)),
            np.hstack((np.zeros((self.ny, self.nx)), np.eye(self.ny)))
        ))

        # B_hat = [B; 0]
        self.B_hat = np.vstack((self.B, np.zeros((self.nd, self.nu))))

        # C_hat = [C  Cd]
        self.C_hat = np.hstack((C, np.ones((self.ny,self.nd))))

        from scipy.signal import place_poles
        res = place_poles(self.A_hat.T, self.C_hat.T, poles)
        self.L = -res.gain_matrix.T

        self.x_hat = np.ndarray((1, self.nx))
        self.d_estimate = np.ndarray((1, self.nd))

        # YOUR CODE HERE
        ##################################################

    def update_estimator(self, x_data: np.ndarray, u_data: np.ndarray) -> None:
        # FOR PART 5 OF THE PROJECT
        ##################################################
        # YOUR CODE HERE
        tmp = self.A_hat @ np.concatenate((self.x_hat, self.d_estimate)) + self.B_hat @ u_data + self.L @ (self.C_hat @ self.x_hat + self.d_gain * self.d_estimate - self.C_hat @ x_data)
        self.x_hat = tmp[:self.nx]
        self.d_estimate = tmp[self.nx:]
        # YOUR CODE HERE
        ##################################################
