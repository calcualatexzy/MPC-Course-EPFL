import numpy as np

from .MPCControl_base import MPCControl_base
from mpt4py import Polyhedron


class MPCControl_yvel(MPCControl_base):
    x_ids: np.ndarray = np.array([0, 3, 7])
    u_ids: np.ndarray = np.array([0])

    def _setup_controller(self) -> None:
        """
        x: [omega_x, alpha, v_y]
        u: delta_1
        """
        #################################################
        # YOUR CODE HERE
        print("setting up yvel")
        self.Q = np.diag([10, 1, 1])
        self.R = np.diag([1])

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

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        #################################################
        # YOUR CODE HERE
        

        # YOUR CODE HERE
        #################################################

        return super(MPCControl_yvel, self).get_u(x0, x_target, u_target)
