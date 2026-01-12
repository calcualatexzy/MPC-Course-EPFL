import numpy as np

from .MPCControl_base import MPCControl_base
from mpt4py import Polyhedron


class MPCControl_xvel(MPCControl_base):
    x_ids: np.ndarray = np.array([1, 4, 6])
    u_ids: np.ndarray = np.array([1])

    def _setup_controller(self) -> None:
        """
        x: [omega_y, beta, v_x]
        u: delta_2
        """
        #################################################
        # YOUR CODE HERE
        print("setting up xvel")
        self.Q = np.diag([10, 1, 1])
        self.R = np.diag([1])
        
        # Input constraints: u in U = { u | Mu <= m }
        # delta_2: -0.26 <= delta_2 <= 0.26
        self.M = np.array([[1], [-1]])  # [delta_2 <= delta_max, -delta_2 <= -delta_min]
        self.m = np.array([self.params['delta'][1], -self.params['delta'][0]])
        self.U = Polyhedron.from_Hrep(self.M, self.m)
        
        # State constraints: x in X = { x | Fx <= f }
        # beta: -0.1745 <= beta <= 0.1745
        self.F = np.array([
            [0, 1, 0],   # beta <= beta_max
            [0, -1, 0],  # -beta <= beta_max (i.e., beta >= -beta_max)
        ])
        self.f = np.array([self.params['beta'][1], -self.params['beta'][0]])
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

        return super(MPCControl_xvel, self).get_u(x0, x_target, u_target)