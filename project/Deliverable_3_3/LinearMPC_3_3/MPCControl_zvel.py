import numpy as np

from LinearMPC_3_3.MPCControl_base import MPCControl_base
from mpt4py import Polyhedron
import cvxpy as cp

from typing import Tuple
class MPCControl_zvel(MPCControl_base):
    x_ids: np.ndarray = np.array([8])
    u_ids: np.ndarray = np.array([2])

    # only useful for part 5 of the project
    d_estimate: np.ndarray
    d_gain: float

    """
    x: [v_z]
    u: P_avg
    """

    def _cost_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get cost matrices Q and R.
        """
        Q = np.diag([10])
        R = np.diag([0.5])
        return Q, R

    def _constraints(self) -> tuple[Polyhedron, Polyhedron]:
        """
        Get constraint bounds (X, U).
        """
        # Input constraints: u in U = { u | Mu <= m }
        # P_avg: 40 <= P_avg <= 80
        M = np.array([[1], [-1]])  # [P_avg <= P_avg_max, -P_avg <= -P_avg_min]
        m = np.array([self.params['P_avg'][1] - self.us[0], -self.params['P_avg'][0] + self.us[0]])
        U = Polyhedron.from_Hrep(M, m)
        # State constraints: x in X = { x | Fx <= f }
        # v_z: -np.inf <= v_z <= np.inf
        F = np.array([
            [0],   # v_z <= v_z_max
        ])
        f = np.array([1])
        X = Polyhedron.from_Hrep(F, f)
        return X, U
