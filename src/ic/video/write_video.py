# Test how to write a video from frames.

import cv2
import numpy as np


def test_write_video():
    # Create a VideoWriter object
    out = cv2.VideoWriter(
        'output.mp4',
        cv2.VideoWriter_fourcc(*'mp4v'),  # Codec for .mp4
        5.0,
        (640, 480),
    )

    # Generate some dummy frames and write to video
    for i in range(100):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            frame,
            f'Frame {i}',
            (50, 240),  # origin
            cv2.FONT_HERSHEY_SIMPLEX,  # font family
            1,  # font scale
            (255, 255, 255),  # color
            2,  # thickness
        )
        out.write(frame)

    out.release()
    print("Video written to output.mp4")