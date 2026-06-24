import pathlib

# Directories.
ROOT_DIR = pathlib.Path(__file__).parent.parent.parent
SRC_DIR = ROOT_DIR / 'src'
VAR_DIR = ROOT_DIR / 'var'
IMAGE_DIR = VAR_DIR / 'images'

# YOLO Models and configuration
YOLO_DIR = VAR_DIR / 'yolo'
YOLO_26N = YOLO_DIR / 'yolo26n.pt'
YOLO_RUNS_DIR = VAR_DIR / 'runs'