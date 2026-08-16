"""A caption-free demo clip to narrate over, for the recorded presentation.

`make_demo_video.py` produces a standalone film: burned-in captions, phases held long enough to
read them, 1 minute 55. That is right for something emailed to a supervisor and wrong for a talk,
because the captions compete with the speaker and the length does not fit the slot slide 49 has.

This rebuilds a shorter, silent, caption-free clip from the frames that run already cached, so the
browser does not have to be driven again. Nothing is re-measured and nothing new is claimed: it is
the same capture, cut differently for a different job.

Two changes from the film. No caption bar, because the presenter is the caption. And shorter phase
minimums, because a phase only has to be seen rather than read.

The live deck's slide 49 has an 85-second slot; the 10-minute submission video cannot afford that,
so the length is an argument rather than a constant and each length writes its own file.

    python dissertation/presentation/make_narration_clip.py            85s, for the live deck
    python dissertation/presentation/make_narration_clip.py --seconds 55 --out Demo_Clip_Submission.mp4
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".demo_frames"

FPS = 24
CAPTURE_HZ = 8
HOLD_STATIC = 1.6          # a motionless stretch is cut harder here than in the film
DIFF_THRESHOLD = 0.004

# Only the phases that show the interface doing its job. The repository tour, the counterfactual
# panel and the guide belong to other slides, so they are dropped rather than rushed.
KEEP = ["open", "loaded", "s1", "s2", "s3", "s4", "inspect", "s5", "contrast"]
# Sized so the clip lands on 85 seconds, which is what slide 49 already had. Matching the slot
# means the talk's measured 19:05 does not move, and the narration written for that slide still
# fits over the top of it.
WEIGHTS = {
    "open": 4, "loaded": 4, "s1": 10, "s2": 10, "s3": 10,
    "s4": 10, "inspect": 10, "s5": 17, "contrast": 10,
}


def changed(a: Image.Image, b: Image.Image) -> float:
    from PIL import ImageChops
    small = (160, 90)
    d = ImageChops.difference(a.resize(small), b.resize(small)).convert("L")
    hist = d.histogram()
    moved = sum(hist[24:])
    return moved / float(small[0] * small[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=85.0, help="target clip length")
    ap.add_argument("--out", default="Demo_Clip_Narration.mp4")
    args = ap.parse_args()
    OUT = HERE / args.out
    scale = args.seconds / sum(WEIGHTS.values())
    MIN_SECONDS = {k: v * scale for k, v in WEIGHTS.items()}
    print(f"target {args.seconds:.0f}s -> phase scale {scale:.3f}")

    if not CACHE.exists():
        sys.exit(f"No cached frames at {CACHE}. Run make_demo_video.py first.")

    frames = sorted(CACHE.glob("*.jpg"))
    per_phase: dict[str, list[Image.Image]] = {}
    for f in frames:
        m = re.match(r"\d+_(.+)\.jpg$", f.name)
        if not m:
            continue
        phase = m.group(1)
        if phase in MIN_SECONDS:
            per_phase.setdefault(phase, []).append(Image.open(f).convert("RGB"))

    missing = [p for p in KEEP if p not in per_phase]
    if missing:
        sys.exit(f"Cached capture is missing phases: {missing}")

    timeline: list[Image.Image] = []
    max_static = int(HOLD_STATIC * CAPTURE_HZ)
    for phase in KEEP:
        kept, prev, static = [], None, 0
        for img in per_phase[phase]:
            moved = prev is None or changed(prev, img) > DIFF_THRESHOLD
            prev = img
            if moved:
                static = 0
            else:
                static += 1
                if static > max_static:
                    continue
            kept.append(img)
        block = [img for img in kept for _ in range(3)]        # 8 Hz capture -> 24 fps
        floor = int(MIN_SECONDS[phase] * FPS)
        if len(block) < floor and kept:
            block += [kept[-1]] * (floor - len(block))
        print(f"  {phase:<9} {len(per_phase[phase]):>4} captured -> {len(block)/FPS:>5.1f}s")
        timeline.extend(block)

    print(f"\ntimeline {len(timeline)} frames = {len(timeline)/FPS:.1f}s at {FPS} fps")

    import imageio_ffmpeg
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for i, im in enumerate(timeline):
            im.save(td / f"{i:06d}.jpg", quality=92)
        cmd = [exe, "-y", "-framerate", str(FPS), "-i", str(td / "%06d.jpg"),
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
               "-movflags", "+faststart", str(OUT)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f"ffmpeg failed:\n{r.stderr[-1500:]}")
    print(f"Wrote {OUT}  ({OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
