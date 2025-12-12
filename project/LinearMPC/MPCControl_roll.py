import numpy as np

from .MPCControl_base import MPCControl_base
from mpt4py import Polyhedron


class MPCControl_roll(MPCControl_base):
    x_ids: np.ndarray = np.array([2, 5])
    u_ids: np.ndarray = np.array([3])

    def _setup_controller(self) -> None:
        """
        x: [omega_z, gamma]
        u: P_diff
        """
        #################################################
        # YOUR CODE HERE

        # Input constraints: u in U = { u | Mu <= m }
        # Pdiff: -20 <= Pdiff <= 20
        self.M = np.array([[1], [-1]])  # [Pdiff <= 20, -Pdiff <= 20]
        self.m = np.array([self.params['P_diff'][1], -self.params['P_diff'][0]])
        self.U = Polyhedron.from_Hrep(self.M, self.m)
        
        # State constraints: x in X = { x | Fx <= f }
        # omega_z: -30 <= omega_z <= 30
        # gamma: -pi <= gamma <= pi
        self.F = np.array([
            [0, 0],   # omega_z <= omega_z_max
        ])
        self.f = np.array([1])
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
