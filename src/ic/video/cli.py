import pathlib

import click
import rich

import cv2

from ic.video import models
from ic.video import settings


@click.group()
def cli():
    """Image Check CLI Tool

    Use this to perform various checks on images for BuySell fraud detection.
    """
    pass


@cli.command()
def hello():
    rich.print("Hello, world!")


@cli.command()
@click.argument('video', type=click.Path(exists=True))
def bike_detection(video):
    """Test bike detection on a sample image."""
    model = models.yolo_26n
    model.predict(
        source=video,
        show=True,           # live window with boxes drawn
        classes=[1],         # COCO class 1 = bicycle only
        conf=0.4,
        save=True,
    )


@cli.command()
@click.argument('source', type=click.Path(exists=True))
def annotate_yolo(source):
    """Test annotation of video frames with bounding boxes and verdicts."""
    from ic import pipeline
    source = pathlib.Path(source)
    copy = source.parent / f"{source.stem}_annotated{source.suffix}"
    frame_count = pipeline.annotate_video(source, copy)
    rich.print(f"[green]Annotated video saved as {copy} ({frame_count} frames).[/green]")


@cli.command()
def compare_images():
    """Compare two bike images using ORB keypoint matching.
    """
    from ic import simple
    bike_1_filename = settings.VAR_DIR / 'rockhopper/img_front.jpg'
    bike_2_filename = settings.VAR_DIR / 'rockhopper/img_front.jpg'
    simple.compare_bikes(bike_1_filename, bike_2_filename)


@cli.command(name='pipeline')
def pipeline_cli():
    from ic import pipeline
    listing_images = [
        settings.VAR_DIR / 'rockhopper/img_front.jpg',
        settings.VAR_DIR / 'rockhopper/img_left.jpg',
        settings.VAR_DIR / 'rockhopper/img_right.jpg',
        settings.VAR_DIR / 'rockhopper/bike-2019-haibike-02.jpg',
        settings.VAR_DIR / 'rockhopper/orange.jpg',
    ]
    for listing_path in listing_images[1:2]:
        result = pipeline.verify_video_against_listing(
            video_path=settings.VAR_DIR / 'rockhopper/vid_01.mov',
            listing_path=listing_path,
            target_fps=2,
            max_frames=40,
            save_crops=True,
            output_dir=settings.VAR_DIR / 'rockhopper/frames',
            generate_demo_video=True,
            demo_video_path=settings.VAR_DIR / 'rockhopper/demo_output.mp4',
            running_verdict=True,
        )
        pipeline.print_report(result)


@cli.command()
def annotate():
    """Test annotation of video frames with bounding boxes and verdicts."""
    from ic import pipeline
    video_path = settings.VAR_DIR / 'rockhopper/vid_01.mov'
    listing_path = settings.VAR_DIR / 'rockhopper/img_front.jpg'
    frame = pipeline.annotate_video(video_path, listing_path)
    if frame is not None:
        cv2.imwrite("annotated_frame.jpg", frame)
        rich.print("[green]Annotated frame saved as annotated_frame.jpg[/green]")
    else:
        rich.print("[red]Failed to annotate frame.[/red]")


@cli.command()
def write_video():
    """Test writing a video from generated frames."""
    from ic import write_video
    write_video.test_write_video()


@cli.command()
def webdemo():
    """Launch the Gradio web demo."""
    from ic import webdemo
    webdemo.launch()


@cli.command()
@click.argument('files', nargs=-1, type=click.Path(exists=True))
def exif(files):
    """Extract and display EXIF metadata from the given files.

    Globs are allowed.
    """
    from ic import exif
    metadata = exif.get_exif(files)
    for file_meta in metadata:
        rich.print(f"[bold]{file_meta['SourceFile']}[/bold]")
        for key, value in file_meta.items():
            if key != 'SourceFile':
                rich.print(f"  [cyan]{key}[/cyan]: {value}")
        rich.print("\n")



@cli.command()
@click.argument('filename', type=click.Path(exists=True))
def parallax(filename):
    """Test homography fit error between consecutive video frames."""
    from ic import parallax

    parallax.run(filename)



@cli.command()
@click.argument('video', type=click.Path(exists=True))
@click.argument('listing', type=click.Path(exists=True))
def verify(video, listing):
    """Run the full verification pipeline on a video and listing image."""
    from ic.claude import pipeline

    result = pipeline.verify(
        video,
        listing,
    )
    annotated_video_path = "demo_output.mp4"  # produced by your annotation code
    print(f"Verdict: {result.verdict}")
    print(f"Aggregate Scores: {result.aggregate}")
    print(f"Annotated video saved to: {annotated_video_path}")
