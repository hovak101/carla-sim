"""Constants shared by collect_training, train and run_inference."""

# --- camera / image geometry ---
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FOV = 90
CAMERA_X = 1.5  # mounted on the hood/windshield area, looking straight ahead
CAMERA_Z = 2.4
CROP_TOP_FRACTION = 0.5  # fraction of the image height removed from the top (the sky)

# size of the network input after cropping
INPUT_HEIGHT = CAMERA_HEIGHT - int(CAMERA_HEIGHT * CROP_TOP_FRACTION)
INPUT_WIDTH = CAMERA_WIDTH

# --- simulation setup ---
CARLA_HOST = "localhost"
CARLA_PORT = 2000
VEHICLE_TYPE = "vehicle.tesla.model3"
SPAWN_INDEX = 5

# --- speed controller ---
MAX_SPEED = 300
TARGET_SPEED = 10
STABLE_TOLERANCE = 1  # recording starts once |speed - TARGET_SPEED| <= this
GAIN = 1.0  # scales how aggressively throttle responds to speed error
DAMP = 0.5  # damping on speed change: raise to reduce oscillation, lower for a faster rise

# --- lane following (data collection) ---
LOOKAHEAD = 5.0  # meters ahead of the car to aim at

# --- data / model files ---
DATA_DIR = "training_data"  # images/<id>.png plus steering.csv, linked by the id column
MODEL_PATH = "steering_model.pt"
