from collections import deque


class SlidingWindowAggregator:
    def __init__(self, window_size: int):
        self.window_size = max(1, window_size)
        self.buffer = deque(maxlen=self.window_size)

    def update(self, features: dict) -> dict:
        self.buffer.append(features)
        keys = self.buffer[0].keys()
        return {k: sum(row[k] for row in self.buffer) / len(self.buffer) for k in keys}
