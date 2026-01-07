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

        # Offset-free MPC: compute steady-state target (x_s, u_s) that compensates for disturbance
        # Solve: [I-A  -B] [x_s]   [-Bd * d]
        #        [C    0 ] [u_s] = [y_ref  ]
        # where y_ref = C @ x_target (we want output to track reference)
        if hasattr(self, 'd_estimate') and self.d_estimate is not None:
            d = self.d_estimate.flatten()
            Bd = self.B  # Bd = B in this system
            C = np.eye(self.nx)  # Full state feedback (ny = nx)
            y_ref = C @ x_target  # Reference output
            
            # Build and solve the target calculation problem
            # [I-A  -B] [x_s]   [-Bd * d]
            # [C    0 ] [u_s] = [y_ref  ]
            M = np.block([
                [np.eye(self.nx) - self.A, -self.B],
                [C, np.zeros((self.nx, self.nu))]
            ])
            rhs = np.concatenate([-Bd @ d, y_ref])
            
            target = np.linalg.solve(M, rhs)
            x_target = target[:self.nx]
            u_target = target[self.nx:]

        self.x0_var.value = x0 - x_target

        # Constraints
        self.constraints = []
        # Initial condition
        self.constraints.append(self.x_var[:, 0] == self.x0_var)
        # System dynamics
        self.constraints.append(self.x_var[:, 1:] == self.A @ self.x_var[:, :-1] + self.B @ self.u_var)

        # State constraints for delta form
        self.constraints.append(self.X.A @ self.x_var[:, :-1] <= (self.X.b - self.X.A @ x_target).reshape(-1, 1))

        # Input constraints for delta form
        self.constraints.append(self.U.A @ self.u_var <= (self.U.b - self.U.A @ u_target).reshape(-1, 1))

        # Terminal Constraints (slightly larger than the original form)
        #KU = Polyhedron.from_Hrep(self.U.A @ self.K, self.U.b - self.U.A @ u_target)

        #X = Polyhedron.from_Hrep(self.X.A, self.X.b - self.X.A @ x_target)

        #self.O_inf = max_invariant_set(self.A_cl, X.intersect(KU))
        #self.constraints.append(self.O_inf.A @ self.x_var[:, -1] <= self.O_inf.b.reshape(-1, 1))

        self.ocp = cp.Problem(cp.Minimize(self.cost), self.constraints)

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

        poles = np.array([0.5, 0.6])

        C = np.ones((self.ny, self.nx))
        Cd = np.ones((self.ny, self.nd))  # Cd for output equation
        Bd = self.B  # Disturbance affects state through same channel as input

        # A_hat = [A  Bd;  0  I]
        self.A_hat = np.vstack((
            np.hstack((self.A, Bd)),
            np.hstack((np.zeros((self.nd, self.nx)), np.eye(self.nd)))
        ))

        # B_hat = [B; 0]
        self.B_hat = np.vstack((self.B, np.zeros((self.nd, self.nu))))

        # C_hat = [C  Cd]
        self.C_hat = np.hstack((C, Cd))

        from scipy.signal import place_poles
        res = place_poles(self.A_hat.T, self.C_hat.T, poles)
        self.L = -res.gain_matrix.T

        # Initialize state estimates as 1D arrays
        self.x_hat = np.zeros((self.nx,))
        self.d_estimate = np.zeros((self.nd,))

        # YOUR CODE HERE
        ##################################################

    def update_estimator(self, x_data: np.ndarray, u_data: np.ndarray) -> None:
        # FOR PART 5 OF THE PROJECT
        ##################################################
        # YOUR CODE HERE
        # x_data is the measured output y (v_z)
        y = x_data.flatten()

        # Augmented state: [x_hat; d_estimate]
        x_aug = np.concatenate((self.x_hat, self.d_estimate))

        # Predicted output: y_hat = C_hat @ x_aug
        y_hat = self.C_hat @ x_aug

        # Luenberger observer update:
        # x_aug_next = A_hat @ x_aug + B_hat @ u + L @ (y - y_hat)
        x_aug_next = self.A_hat @ x_aug + self.B_hat @ u_data.flatten() + self.L @ (y_hat - y)

        self.x_hat = x_aug_next[:self.nx]
        self.d_estimate = x_aug_next[self.nx:]
        # YOUR CODE HERE
        ##################################################
