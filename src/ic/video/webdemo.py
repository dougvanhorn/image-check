import gradio as gr

from ic.video import pipeline


def verify(listing_image, video):
    # your verification function from before
    result = pipeline.verify_video_against_listing(
        video,
        listing_image,
        save_crops=True,
    )
    annotated_video_path = "demo_output.mp4"  # produced by your annotation code
    return annotated_video_path, result.verdict, result.aggregate


def launch():
    demo = gr.Interface(
        fn=verify,
        inputs=[gr.Image(type="filepath"), gr.Video()],
        outputs=[gr.Video(), gr.Text(label="Verdict"), gr.JSON(label="Scores")],
        title="Pinkbike BuySell Verification POC",
    )
    demo.launch()