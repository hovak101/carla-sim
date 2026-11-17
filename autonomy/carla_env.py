"""CARLA world setup shared by data collection and inference."""
import math
import queue

import carla

from . import config


def connect(host: str = config.CARLA_HOST, port: int = config.CARLA_PORT):
    client = carla.Client(host, port)
    world = client.get_world()
    return client, world


def hide_map_layers(world) -> None:
    """Remove everything except the road itself (layers only work on "_Opt" maps like Town10HD_Opt)."""
    # Unload one layer at a time: combining them with `|` yields an int, which the API rejects.
    for layer in [
        carla.MapLayer.Buildings,
        carla.MapLayer.Decals,
        carla.MapLayer.Foliage,
        carla.MapLayer.Ground,
        carla.MapLayer.ParkedVehicles,
        carla.MapLayer.Particles,
        carla.MapLayer.Props,
        carla.MapLayer.StreetLights,
        carla.MapLayer.Walls,
    ]:
        world.unload_map_layer(layer)

    # Traffic lights, traffic signs and the remaining buildings are part of the base level, not toggleable
    # layers, so hide them individually as environment objects.
    ids_to_hide = set()
    for label in [
        carla.CityObjectLabel.TrafficLight,
        carla.CityObjectLabel.TrafficSigns,
        carla.CityObjectLabel.Poles,  # sign / traffic light poles
        carla.CityObjectLabel.Buildings,
    ]:
        ids_to_hide.update(obj.id for obj in world.get_environment_objects(label))
    world.enable_environment_objects(ids_to_hide, False)
    print(f"Hid {len(ids_to_hide)} environment objects")


def spawn_vehicle(world, spawn_index: int, vehicle_type: str = config.VEHICLE_TYPE):
    """Spawn the car at a fixed spawn point (same car + location every run) and park the spectator above it."""
    blueprint = world.get_blueprint_library().find(vehicle_type)
    spawn_point = world.get_map().get_spawn_points()[spawn_index]
    vehicle = world.spawn_actor(blueprint, spawn_point)
    print(f"Spawned {vehicle.type_id} (id {vehicle.id}) at spawn point {spawn_index}: {spawn_point.location}")

    world.get_spectator().set_transform(carla.Transform(
        spawn_point.location + carla.Location(z=30),
        carla.Rotation(pitch=-90),
    ))
    return vehicle


class CameraRig:
    """Forward facing RGB camera attached to the vehicle; frames are handed to the main loop through a queue."""

    def __init__(self, world, vehicle):
        blueprint = world.get_blueprint_library().find("sensor.camera.rgb")
        blueprint.set_attribute("image_size_x", str(config.CAMERA_WIDTH))
        blueprint.set_attribute("image_size_y", str(config.CAMERA_HEIGHT))
        blueprint.set_attribute("fov", str(config.CAMERA_FOV))
        transform = carla.Transform(carla.Location(x=config.CAMERA_X, z=config.CAMERA_Z))
        self.camera = world.spawn_actor(blueprint, transform, attach_to=vehicle)
        # the callback runs on another thread
        self._queue = queue.Queue()
        self.camera.listen(self._queue.put)
        print(f"Spawned camera (id {self.camera.id}) attached to vehicle {vehicle.id}")

    def clear(self) -> None:
        """Drop any queued frames (e.g. captured before a run starts)."""
        self.drain()

    def drain(self) -> list:
        """All frames received since the last call (oldest first); never blocks."""
        images = []
        while not self._queue.empty():
            images.append(self._queue.get_nowait())
        return images

    def latest(self, timeout: float = 2.0):
        """The newest frame: waits up to `timeout` s for one if the queue is empty, then skips any older ones."""
        image = self._queue.get(timeout=timeout)
        while not self._queue.empty():
            image = self._queue.get_nowait()
        return image

    def destroy(self) -> None:
        self.camera.stop()
        self.camera.destroy()


class SpeedController:
    """Throttle controller that holds the vehicle at `target` speed (same units as `speed`)."""

    def __init__(self, vehicle, target: float = config.TARGET_SPEED, max_speed: float = config.MAX_SPEED,
                 gain: float = config.GAIN, damp: float = config.DAMP):
        self.vehicle = vehicle
        self.target = target
        self.max_speed = max_speed
        self.gain = gain
        self.damp = damp
        self.throttle = 0.0
        self.speed = self._read_speed()
        self._prev_speed = self.speed

    def _read_speed(self) -> float:
        v = self.vehicle.get_velocity()
        return math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)

    def step(self) -> float:
        """Update from the current speed and return the throttle in [0, 1]. Call once per simulator tick."""
        self.speed = self._read_speed()
        self.throttle += (self.gain * (self.target - self.speed) / self.max_speed
                          - self.damp * (self.speed - self._prev_speed))
        self._prev_speed = self.speed
        self.throttle = max(0.0, min(1.0, self.throttle))
        return self.throttle

    def is_stable(self, tolerance: float = config.STABLE_TOLERANCE) -> bool:
        return abs(self.speed - self.target) <= tolerance
