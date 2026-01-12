import numpy as np
import casadi as ca
from typing import Tuple
from control import dlqr

class NmpcCtrl:
    """
    Nonlinear MPC controller for rocket landing.

    Uses CasADi with IPOPT solver for full nonlinear optimization.
    No subsystem decomposition - controls all 12 states and 4 inputs simultaneously.
    """

    def __init__(self, rocket, Ts=0.1, H=2.0, xs=None, us=None):
        """
        Initialize nonlinear MPC controller.

        Args:
            rocket: Rocket object with symbolic dynamics
            Ts: Sampling time (default: 0.1s)
            H: Prediction horizon (default: 2.0s)
            xs: Trim state (optional, for default reference)
            us: Trim input (optional, for default reference)
        """
        # symbolic dynamics f(x,u) from rocket
        self.f = lambda x, u: rocket.f_symbolic(x, u)[0]

        self.Ts = Ts
        self.H = H
        self.N = int(H / Ts)

        # State and input dimensions
        self.nx = 12  # [wx, wy, wz, alpha, beta, gamma, vx, vy, vz, x, y, z]
        self.nu = 4   # [d1, d2, Pavg, Pdiff]

        # Store trim values for default references
        self.xs = xs if xs is not None else np.zeros(self.nx)
        self.us = us if us is not None else np.array([0.0, 0.0, 56.67, 0.0])

        # Reference target (will be set in get_u)
        self.x_ref = np.zeros(self.nx)
        self.u_ref = np.zeros(self.nu)

        # Store last successful solution for fallback
        self.last_u0 = self.us.copy()
        self.A, self.B = rocket.linearize(self.xs, self.us)

        # Setup the optimization problem
        self._setup_controller(xs, us)

    def _setup_controller(self, xs=None, us=None) -> None:
        """
        Setup nonlinear MPC optimization problem using CasADi.

        Uses direct multiple shooting with RK4 integration.
        """
        # Decision variables
        X = ca.MX.sym('X', self.nx, self.N + 1)  # States over horizon
        U = ca.MX.sym('U', self.nu, self.N)       # Inputs over horizon

        # Parameters (initial state and reference)
        x0 = ca.MX.sym('x0', self.nx)
        x_ref = ca.MX.sym('x_ref', self.nx)
        u_ref = ca.MX.sym('u_ref', self.nu)

        # Cost matrices
        Q = ca.diag(ca.vertcat(
            1.0, 1.0, 1.0,          # angular velocities (wx, wy, wz)
            10.0, 10.0, 15.0,       # angles (alpha, beta, gamma)
            1.0, 1.0, 10.0,         # velocities (vx, vy, vz) - higher vz weight for safe landing
            10.0, 10.0, 15.0        # positions (x, y, z) - higher z weight for altitude tracking
        ))

        # Input cost: penalize input effort
        R = ca.diag(ca.vertcat(0.1, 0.1, 0.1, 0.1))  # d1, d2, Pavg, Pdiff

        Q_np = np.array(Q.full())
        R_np = np.array(R.full())
        _, P, _ = dlqr(self.A, self.B, Q_np, R_np)
        P = ca.DM(P)

        # Build the cost function
        cost = 0
        for k in range(self.N):
            dx = X[:, k] - x_ref
            du = U[:, k] - u_ref
            cost += dx.T @ Q @ dx + du.T @ R @ du

        # Terminal cost
        dx_N = X[:, self.N] - x_ref
        cost += dx_N.T @ P @ dx_N

        # Build constraints
        g = []  # Constraint vector
        lbg = []  # Lower bounds on constraints
        ubg = []  # Upper bounds on constraints

        # Initial condition constraint
        g.append(X[:, 0] - x0)
        lbg.extend([0.0] * self.nx)
        ubg.extend([0.0] * self.nx)

        # Dynamics constraints (using RK4 integration for better accuracy)
        for k in range(self.N):
            # RK4: x_{k+1} = x_k + (Ts/6) * (k1 + 2*k2 + 2*k3 + k4)
            k1 = self.f(X[:, k], U[:, k])
            k2 = self.f(X[:, k] + self.Ts / 2 * k1, U[:, k])
            k3 = self.f(X[:, k] + self.Ts / 2 * k2, U[:, k])
            k4 = self.f(X[:, k] + self.Ts * k3, U[:, k])
            x_next = X[:, k] + self.Ts / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            g.append(X[:, k + 1] - x_next)
            lbg.extend([0.0] * self.nx)
            ubg.extend([0.0] * self.nx)

        # State constraints
        # 1. z >= 0 (don't go underground)
        # 2. |beta| <= 80° (avoid singularity at beta = 90°)
        for k in range(self.N + 1):
            # z >= 0
            g.append(X[11, k])  # z (12th state, index 11)
            lbg.append(0.0)     # z >= 0
            ubg.append(ca.inf)  # No upper bound

            # |beta| <= 80° (beta is state index 4)
            g.append(X[4, k])   # beta (5th state, index 4)
            lbg.append(-np.deg2rad(80))  # beta >= -80°
            ubg.append(np.deg2rad(80))   # beta <= 80°

        # Concatenate constraints
        g = ca.vertcat(*g)

        # Decision variable bounds
        # State bounds (mostly unbounded except z >= 0 which is in constraints)
        lbx = []
        ubx = []

        for k in range(self.N + 1):
            # States: unbounded except constraints above
            lbx.extend([-ca.inf] * self.nx)
            ubx.extend([ca.inf] * self.nx)

        for k in range(self.N):
            # Input bounds (NMPC has wider Pavg bounds than linear MPC)
            # IMPORTANT: d1, d2 are in RADIANS, Pavg is percent, d3 is DEGREES
            lbx.extend([np.deg2rad(-15), np.deg2rad(-15), 10.0, -20.0])  # d1, d2, Pavg, d3
            ubx.extend([np.deg2rad(15), np.deg2rad(15), 90.0, 20.0])

        # Decision variables vector
        w = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))

        # Parameters vector
        p = ca.vertcat(x0, x_ref, u_ref)

        # Create NLP solver
        nlp = {
            'x': w,
            'f': cost,
            'g': g,
            'p': p
        }

        opts = {
            'ipopt.print_level': 0,
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.max_iter': 100,
            'print_time': 0,
            'ipopt.acceptable_tol': 1e-6,
            'ipopt.acceptable_obj_change_tol': 1e-6
        }

        self.solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        # Store problem dimensions for solution extraction
        self.lbx = lbx
        self.ubx = ubx
        self.lbg = lbg
        self.ubg = ubg

        # Store variable shapes
        self.X_shape = (self.nx, self.N + 1)
        self.U_shape = (self.nu, self.N)

        # Initialize solution with zeros
        self.w0 = np.zeros((self.nx * (self.N + 1) + self.nu * self.N, 1))

    def get_u(
        self, t0: float, x0: np.ndarray, x_ref: np.ndarray = None, u_ref: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve nonlinear MPC and return control input.

        Args:
            t0: Current time
            x0: Current state (12,)
            x_ref: Reference state (12,) - optional, defaults to self.xs
            u_ref: Reference input (4,) - optional, defaults to self.us

        Returns:
            u0: Optimal control input (4,)
            x_ol: Predicted state trajectory (12, N+1)
            u_ol: Predicted input trajectory (4, N)
            t_ol: Time vector (N+1,)
        """
        # Set references (use stored trim values if not provided)
        if x_ref is None:
            x_ref = self.xs
        if u_ref is None:
            u_ref = self.us

        # Parameters
        p = np.concatenate([x0, x_ref, u_ref])

        # Solve NLP
        
        sol = self.solver(
            x0=self.w0,
            lbx=self.lbx,
            ubx=self.ubx,
            lbg=self.lbg,
            ubg=self.ubg,
            p=p
        )

        # Extract solution
        w_opt = sol['x'].full().flatten()

        # Warm start for next iteration
        self.w0 = w_opt.reshape(-1, 1)

        # Parse solution
        X_opt = w_opt[:self.nx * (self.N + 1)].reshape(self.nx, self.N + 1, order='F')
        U_opt = w_opt[self.nx * (self.N + 1):].reshape(self.nu, self.N, order='F')

        # First control input
        u0 = U_opt[:, 0]

        # Store successful solution for fallback
        self.last_u0 = u0.copy()

        # Trajectories
        x_ol = X_opt
        u_ol = U_opt
        t_ol = np.arange(self.N + 1) * self.Ts + t0

        return u0, x_ol, u_ol, t_ol
