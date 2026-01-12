import numpy as np

from LinearMPC_5_2.MPCControl_base import MPCControl_base
from mpt4py import Polyhedron


class MPCControl_roll(MPCControl_base):
    x_ids: np.ndarray = np.array([2, 5])
    u_ids: np.ndarray = np.array([3])
    """
    x: [omega_z, gamma]
    u: P_diff
    """

    def _cost_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get cost matrices Q and R.
        """
        Q = np.diag([1, 20])
        R = np.diag([1])
        return Q, R

    def _constraints(self) -> tuple[Polyhedron, Polyhedron]:
        """
        Get constraint bounds (X, U).
        """
        # Input constraints: u in U = { u | Mu <= m }
        # Pdiff: -20 <= Pdiff <= 20
        M = np.array([[1], [-1]])  # [Pdiff <= 20, -Pdiff <= 20]
        m = np.array([self.params['P_diff'][1] - self.us[0], -self.params['P_diff'][0] + self.us[0]])
        U = Polyhedron.from_Hrep(M, m)
        
        # State constraints: x in X = { x | Fx <= f }
        # omega_z: -30 <= omega_z <= 30
        # gamma: -pi <= gamma <= pi
        F = np.array([
            [0, 0],   # omega_z <= omega_z_max
        ])
        f = np.array([1])
        X = Polyhedron.from_Hrep(F, f)
        
        return X, U

    def _C(self) -> np.ndarray:
        """
        Get the measure matrix C.
        """
        return np.array([[0.0, 1.0]])