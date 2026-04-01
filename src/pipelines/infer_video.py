import argparse

import cv2
from tqdm import tqdm

from src.core.settings import load_runtime_settings
from src.io.frame_sampler import should_keep_frame
from src.io.video_reader import VideoReader
from src.io.writer import VideoWriter, write_csv
from src.pipelines.infer_image import ImageInferencePipeline
from src.pipelines.postprocess import SlidingWindowAggregator
from src.viz.dashboard import plot_timeline
from src.viz.overlay import draw_overlay
from src.rules.state_mapper import map_state


def run_video(input_video: str, output_video: str, output_csv: str, output_figure: str | None = None):
    runtime, cfg = load_runtime_settings()
    pipe = ImageInferencePipeline()
    vr = VideoReader(input_video)
    agg = SlidingWindowAggregator(runtime.window_sec * runtime.fps_output)

    rows = []
    writer = VideoWriter(output_video, runtime.resize_width, runtime.resize_height, runtime.fps_output)
    for idx, frame, ts in tqdm(vr.read(), desc="infer_video"):
        if not should_keep_frame(idx, runtime.fps_input, runtime.fps_output):
            continue
        frame = cv2.resize(frame, (runtime.resize_width, runtime.resize_height))
        state, conf, scores, feats = pipe.run_frame(frame)
        merged = agg.update({**scores, **feats})
        state, conf = map_state(
            merged["attention_score"], merged["affect_score"], merged["blocked_score"], merged["face_presence"]
        )
        rows.append({"timestamp": ts, "state": state, "confidence": conf, **merged})
        writer.write(draw_overlay(frame, state, conf, scores, feats))
    writer.close()
    write_csv(rows, output_csv)
    if output_figure:
        plot_timeline(output_csv, output_figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/demo_videos/sample.mp4")
    parser.add_argument("--out_video", default="data/outputs/videos/demo_overlay.mp4")
    parser.add_argument("--out_csv", default="data/outputs/csv/demo_states.csv")
    parser.add_argument("--out_figure", default="data/outputs/figures/demo_timeline.png")
    args = parser.parse_args()
    run_video(args.input, args.out_video, args.out_csv, args.out_figure)


if __name__ == "__main__":
    main()
