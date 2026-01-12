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
        print("setting up roll")
        self.Q = np.diag([1, 20])
        self.R = np.diag([1])

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

    def get_u(
        self, x0: np.ndarray, x_target: np.ndarray = None, u_target: np.ndarray = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        #################################################
        # YOUR CODE HERE
        

        # YOUR CODE HERE
        #################################################

        return super(MPCControl_roll, self).get_u(x0, x_target, u_target)
