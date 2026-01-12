import numpy as np

from LinearMPC_4_1.MPCControl_base import MPCControl_base
from mpt4py import Polyhedron


class MPCControl_xvel(MPCControl_base):
    x_ids: np.ndarray = np.array([1, 4, 6])
    u_ids: np.ndarray = np.array([1])
    """
    x: [omega_y, beta, v_x]
    u: delta_2
    """

    def _cost_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get cost matrices Q and R.
        """
        Q = np.diag([100, 1, 1])
        R = np.diag([0.5])
        return Q, R

    def _constraints(self) -> tuple[Polyhedron, Polyhedron]:
        """
        Get constraint bounds (X, U).
        """
        # Input constraints: u in U = { u | Mu <= m }
        # delta_2: -0.26 <= delta_2 <= 0.26
        M = np.array([[1], [-1]])  # [delta_2 <= 0.26, -delta_2 <= 0.26]
        m = np.array([self.params['delta'][1] - self.us[0], -self.params['delta'][0] + self.us[0]])
        U = Polyhedron.from_Hrep(M, m)

        # State constraints: x in X = { x | Fx <= f }
        # beta: -0.1745 <= beta <= 0.1745
        F = np.array([
            [0, 1, 0],   # beta <= beta_max
            [0, -1, 0],  # -beta <= beta_max (i.e., beta >= -beta_max)
        ])
        f = np.array([self.params['beta'][1], -self.params['beta'][0]])
        X = Polyhedron.from_Hrep(F, f)
        
        return X, U

    def _C(self) -> np.ndarray:
        """
        Get the measure matrix C.
        """
        return np.array([[0.0, 0.0, 1.0]])