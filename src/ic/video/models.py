# Load models one time here.
from ultralytics import YOLO

from ic import settings

# ----- Models loaded once at module level -----
_yolo_26n = YOLO(settings.YOLO_26N)
def _load_yolo_26n():
    global _yolo_26n
    if _yolo_26n is None:
        _yolo_26n = YOLO(settings.YOLO_26N)
    return _yolo_26n

# print("Loading CLIP...")
# clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
#     'ViT-B-32',
#     pretrained='openai'
# )
# clip_model.eval()


def __getattr__(name):
    if name == 'yolo_26n':
        return _load_yolo_26n()

    raise AttributeError(f"Module {__name__} has no attribute {name}")