class AggressiveBrakingTrajectory:
    """
    Forward motion followed by abrupt braking.
    This trajectory excites longitudinal payload motion.
    """

    def __init__(self, speed=1.5, brake_time=1.5):
        self.speed = speed
        self.brake_time = brake_time
        self.stop_position = speed * brake_time

    def get_desired_state(self, t):
        if t < self.brake_time:
            x = self.speed * t
            vx = self.speed
        else:
            x = self.stop_position
            vx = 0.0

        return {
            "x": x,
            "y": 0.0,
            "yaw": 0.0,
            "vx": vx,
            "vy": 0.0,
            "yaw_rate": 0.0,
        }