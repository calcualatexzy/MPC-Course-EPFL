import numpy as np

from LinearMPC_3_2.MPCControl_base import MPCControl_base
from mpt4py import Polyhedron


class MPCControl_yvel(MPCControl_base):
    x_ids: np.ndarray = np.array([0, 3, 7])
    u_ids: np.ndarray = np.array([0])

    """
    x: [omega_x, alpha, v_y]
    u: delta_1
    """

    def _cost_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get cost matrices Q and R.
        """
        Q = np.diag([10, 1, 1])
        R = np.diag([1])
        return Q, R

    def _constraints(self) -> tuple[Polyhedron, Polyhedron]:

        # Input constraints: u in U = { u | Mu <= m }
        # delta_1: -0.26 <= delta_1 <= 0.26
        M = np.array([[1], [-1]])  # [delta_1 <= delta_max, -delta_1 <= delta_max]
        m = np.array([self.params['delta'][1] - self.us[0], -self.params['delta'][0] + self.us[0]])
        U = Polyhedron.from_Hrep(M, m)
        
        # State constraints: x in X = { x | Fx <= f }
        # alpha: -0.1745 <= alpha <= 0.1745
        F = np.array([
            [0, 1, 0],   # alpha <= alpha_max
            [0, -1, 0],  # -alpha <= alpha_max (i.e., alpha >= -alpha_max)
        ])
        f = np.array([self.params['alpha'][1], -self.params['alpha'][0]])
        X = Polyhedron.from_Hrep(F, f)
        return X, U