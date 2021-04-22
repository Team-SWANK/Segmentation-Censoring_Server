# initialize Redis connection settings
REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

# initialize constants used to control image spatial dimensions and
# data type
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_CHANS = 3
IMAGE_DTYPE = "float32"

# initialize constants used for server queuing
SEGMENT_IMAGE_QUEUE = "seg_image_queue"
CENSOR_IMAGE_QUEUE = "cens_image_queue"
BATCH_SIZE = 10
SERVER_SLEEP = 0.25
CLIENT_SLEEP = 0.25