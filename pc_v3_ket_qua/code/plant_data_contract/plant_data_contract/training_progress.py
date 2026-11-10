"""One in-place training indicator, with compact summaries for redirected stdout."""

import shutil
import sys
import time


class TrainingProgress:
    def __init__(self, total_steps, stream=None, interactive=None):
        self.stream = stream if stream is not None else sys.stdout
        self.interactive = bool(self.stream.isatty()) if interactive is None else bool(interactive)
        self.total_steps = max(1, int(total_steps))
        self.completed = 0
        self.started = time.monotonic()
        self.last_width = 0
        self.last_render = 0.0
        self.closed = False

    def update(self, *, increment=0, phase="train", position="", loss=None, validation=None, lr=None, vram_mb=None):
        if self.closed:
            raise RuntimeError("Training progress was already closed")
        self.completed = min(self.total_steps, self.completed + increment)
        if not self.interactive:
            return
        now = time.monotonic()
        if self.completed < self.total_steps and now - self.last_render < 0.2:
            return
        self.last_render = now
        elapsed = max(0.0, now - self.started)
        eta = elapsed * (self.total_steps - self.completed) / self.completed if self.completed else 0.0
        ratio = self.completed / self.total_steps
        width = max(8, min(24, shutil.get_terminal_size((100, 20)).columns - 76))
        filled = round(width * ratio)
        fields = [f"[{('=' * filled):<{width}}]", f"{ratio:5.1%}", position, phase, f"ETA {eta:.0f}s"]
        if loss is not None:
            fields.append(f"loss {loss:.4f}")
        if validation is not None:
            fields.append(f"PD-F1 {validation:.4f}")
        if lr is not None:
            fields.append(f"LR {lr:.2g}")
        if vram_mb is not None:
            fields.append(f"VRAM {vram_mb:.0f}MB")
        columns = max(20, shutil.get_terminal_size((100, 20)).columns)
        line = " ".join(fields).replace("\r", " ").replace("\n", " ")[:columns - 1]
        self.stream.write("\r" + line + " " * max(0, self.last_width - len(line)))
        self.stream.flush()
        self.last_width = len(line)

    def summary(self, text):
        if self.interactive and self.last_width:
            self.stream.write("\r" + " " * self.last_width + "\r")
            self.last_width = 0
        self.stream.write(text + "\n")
        self.stream.flush()

    def close(self):
        if not self.closed:
            if self.interactive and self.last_width:
                self.stream.write("\n")
                self.stream.flush()
            self.closed = True
