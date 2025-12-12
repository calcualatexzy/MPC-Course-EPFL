import numpy as np

from .MPCControl_base import MPCControl_base
from mpt4py import Polyhedron


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

    def setup_estimator(self):
        # FOR PART 5 OF THE PROJECT
        ##################################################
        # YOUR CODE HERE

        self.d_estimate = ...
        self.d_gain = ...

        # YOUR CODE HERE
        ##################################################

    def update_estimator(self, x_data: np.ndarray, u_data: np.ndarray) -> None:
        # FOR PART 5 OF THE PROJECT
        ##################################################
        # YOUR CODE HERE
        self.d_estimate = ...
        # YOUR CODE HERE
        ##################################################
