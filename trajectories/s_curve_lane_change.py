import numpy as np


class SCurveLaneChangeTrajectory:
    """
    Smooth S-curve lane-change trajectory.

    The robot moves forward while shifting laterally, then returns.
    This creates lateral acceleration that should excite side-to-side payload motion.
    """

    def __init__(self, speed=0.6, amplitude=0.45, duration=8.0):
        self.speed = speed
        self.amplitude = amplitude
        self.duration = duration

    def get_desired_state(self, t):
        T = self.duration
        A = self.amplitude
        v = self.speed

        # Clamp time after trajectory ends
        tau = min(t, T)

        x = v * tau
        vx = v

        # Smooth S-curve: starts at 0, moves left, returns to 0
        y = A * np.sin(2.0 * np.pi * tau / T)
        vy = A * (2.0 * np.pi / T) * np.cos(2.0 * np.pi * tau / T)

        # Desired yaw points along velocity direction
        yaw = np.arctan2(vy, vx)

        # Approximate yaw rate numerically/analytically
        ay = -A * (2.0 * np.pi / T) ** 2 * np.sin(2.0 * np.pi * tau / T)
        yaw_rate = (vx * ay) / (vx ** 2 + vy ** 2)

        if t > T:
            y = 0.0
            vy = 0.0
            yaw = 0.0
            yaw_rate = 0.0

        return {
            "x": x,
            "y": y,
            "yaw": yaw,
            "vx": vx,
            "vy": vy,
            "yaw_rate": yaw_rate,
        }