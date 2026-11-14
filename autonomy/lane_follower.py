"""Waypoint-based steering used as the "expert" while collecting training data."""
import math

from . import config


class WaypointSteering:
    def __init__(self, vehicle, carla_map, lookahead: float = config.LOOKAHEAD):
        self.vehicle = vehicle
        self.carla_map = carla_map
        self.lookahead = lookahead
        # normalize by the front wheels' max steer angle (degrees) -> steer in [-1, 1]
        self.max_steer = vehicle.get_physics_control().wheels[0].max_steer_angle

    def steer(self) -> float:
        """VehicleControl.steer in [-1, 1] that points the car at the leftmost lane waypoint `lookahead` m ahead."""
        waypoint = self.carla_map.get_waypoint(self.vehicle.get_location())

        # waypoints ahead (next() returns a list; at junctions it has several branches)
        candidates = waypoint.next(self.lookahead)
        if not isinstance(candidates, (list, tuple)):
            candidates = [candidates]
        transform = self.vehicle.get_transform()

        def heading_error(w):
            # signed angle (deg) from the vehicle's heading to waypoint w.
            # CARLA yaw is clockwise-positive, so positive = target is to the right, negative = to the left
            loc = w.transform.location
            desired_yaw = math.degrees(math.atan2(loc.y - transform.location.y, loc.x - transform.location.x))
            error = desired_yaw - transform.rotation.yaw
            return (error + 180) % 360 - 180  # wrap to [-180, 180]

        # leftmost waypoint = the one with the smallest (most negative) signed angle
        error = min(heading_error(w) for w in candidates)
        return max(-1.0, min(1.0, error / self.max_steer))
