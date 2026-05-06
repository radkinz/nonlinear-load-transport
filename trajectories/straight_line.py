class StraightLineTrajectory:
    def __init__(self, speed=0.25, target_x=1.2):
        self.speed = speed
        self.target_x = target_x
        self.move_time = target_x / speed

    def get_desired_state(self, t):
        if t < self.move_time:
            x = self.speed * t
            vx = self.speed
        else:
            x = self.target_x
            vx = 0.0

        return {
            "x": x,
            "y": 0.0,
            "yaw": 0.0,
            "vx": vx,
            "vy": 0.0,
            "yaw_rate": 0.0,
        }