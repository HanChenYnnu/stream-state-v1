
def should_keep_frame(frame_idx: int, fps_input: int, fps_output: int) -> bool:
    if fps_output >= fps_input:
        return True
    step = max(1, int(round(fps_input / fps_output)))
    return frame_idx % step == 0
