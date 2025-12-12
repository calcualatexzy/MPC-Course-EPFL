import numpy as np

from .MPCControl_base import MPCControl_base
from mpt4py import Polyhedron


class MPCControl_yvel(MPCControl_base):
    x_ids: np.ndarray = np.array([0, 3, 7])
    u_ids: np.ndarray = np.array([0])

    def _setup_controller(self) -> None:
        """
        x: [omega_x, alpha, v_y, y]
        u: delta_1
        """
        #################################################
        # YOUR CODE HERE

        # Input constraints: u in U = { u | Mu <= m }
        # delta_1: -0.26 <= delta_1 <= 0.26
        self.M = np.array([[1], [-1]])  # [delta_1 <= delta_max, -delta_1 <= delta_max]
        self.m = np.array([self.params['delta'][1], -self.params['delta'][0]])
        self.U = Polyhedron.from_Hrep(self.M, self.m)
        
        # State constraints: x in X = { x | Fx <= f }
        # alpha: -0.1745 <= alpha <= 0.1745
        self.F = np.array([
            [0, 1, 0],   # alpha <= alpha_max
            [0, -1, 0],  # -alpha <= alpha_max (i.e., alpha >= -alpha_max)
        ])
        self.f = np.array([self.params['alpha'][1], -self.params['alpha'][0]])
        self.X = Polyhedron.from_Hrep(self.F, self.f)
    
        super()._setup_controller()

        # YOUR CODE HERE
        #################################################

    # def get_u(
    #     self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    # ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    #     #################################################
    #     # YOUR CODE HERE

    #     if x_target is None:
    #         x_target = self.xs
    #     if u_target is None:
    #         u_target = self.us

    #     self.x0_var.value = x0 - x_target
    #     self.ocp.solve()
    #     u0 = self.u_var.value[:, 0] + u_target
    #     x_traj = self.x_var.value[:, :] + x_target.reshape(-1, 1)
    #     u_traj = self.u_var.value[:, :] + u_target.reshape(-1, 1)

    #     # YOUR CODE HERE
    #     #################################################

    #     return u0, x_traj, u_traj
