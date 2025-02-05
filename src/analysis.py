from statistics import variance
import numpy as np

class Analysis:
    """
    A deterministic, real-time approach for computing a single deception
    probability each time we receive a new gaze point (time, x, y).
    """
    def __init__(self):
        """
        Minimal internal state to compute velocity, acceleration:
         - last_x, last_y, last_time
         - last_velocity
        """
        self.last_x = None
        self.last_y = None
        self.last_time = None
        self.last_velocity = 0.0

    def single_update(self, t, x, y):
        """
        Process one new gaze point at time 't' with coordinates (x,y).
        Returns a single deception probability in [0.01..0.99].
        """
        # If this is the first frame, not enough data for velocity/variance:
        if self.last_time is None:
            self.last_x = x
            self.last_y = y
            self.last_time = t
            self.last_velocity = 0.0
            return (0.0, 0.0, 0.05) # default for first detection

        # Time delta
        dt = t - self.last_time
        if dt <= 0.0:
            return (0.0, 0.0, 0.05) # No forward time => return middle prob

        # 1) "Variance" ~ squared distance to last point
        dx = x - self.last_x
        dy = y - self.last_y
        variance = dx*dx + dy*dy  # This is an approximation of "variance"

        # 2) Velocity
        velocity = np.sqrt(variance) / dt

        # 3) Acceleration => (velocity - last_velocity)/dt
        acceleration = 0.0
        if dt > 0:
            acceleration = (velocity - self.last_velocity) / dt

        # 4) Clip each to [0..10], scale to [0.01..0.99]
        def clip_and_scale(value, MIN, MAX):
            val_abs = abs(value)
            clipped = min(max(val_abs, MIN), MAX)
            return 0.01 + 0.95*(clipped/MAX)

        variance_norm = clip_and_scale(variance, 4.5e-07, 0.00013)
        # velocity_norm = clip_and_scale(velocity, 0.0, 10.0)
        acceleration_norm = clip_and_scale(acceleration, 0.3, 10.0)

        # A simple approach: average the scaled "variance" + scaled "accel"
        # or we could weigh them however we want. We'll do an average of all three.
        # You can pick just variance+acc or variance+velocity, etc.
        probability = (variance_norm + acceleration_norm) / 2.0

        # Update last_x,y,t,velocity
        self.last_x = x
        self.last_y = y
        self.last_time = t
        self.last_velocity = velocity

        return variance_norm, acceleration_norm, probability
    
