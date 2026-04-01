import cv2
import gradio as gr

from src.pipelines.infer_image import ImageInferencePipeline
from src.rules.state_mapper import STATE_TO_ZH

pipe = ImageInferencePipeline()


def infer(image):
    if image is None:
        return "No image"
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    state, conf, scores, feats = pipe.run_frame(frame)
    msg = f"{state} {STATE_TO_ZH[state]}\\nconf={conf:.2f}\\nscores={scores}\\nfeatures={feats}"
    return msg


def main():
    demo = gr.Interface(fn=infer, inputs=gr.Image(type="numpy", source="webcam"), outputs="text", title="K12 Stream State v1")
    demo.launch()


if __name__ == "__main__":
    main()
