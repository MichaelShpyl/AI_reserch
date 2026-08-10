"""Record the demonstration video by driving the real interface.

This produces a captioned screen recording of the actual pipeline running on an actual essay. It
is not a slideshow of screenshots and it is not a mock-up: a headless Chrome loads the app on
127.0.0.1, presses the button, and every frame is what the page really showed at that moment.

Why record it this way rather than with a screen recorder. A hand-driven recording carries the
cursor wandering, the odd mis-click, and a different result every take. This one is deterministic,
it can be re-run after any change to the interface, and it cannot accidentally capture an email
client in the corner of the screen.

The run takes about ninety-five seconds, most of which is the page waiting on a model. Frames are
captured at roughly eight a second and then de-duplicated: stretches where nothing moves are cut
down to a short hold, so the finished video is around three minutes instead of six of watching a
spinner. Nothing is sped up while something is actually happening.

Prerequisites, both of which the script checks:
    ollama serve
    python src/webapp/server.py     (wait for "Ready on http://127.0.0.1:8000")

    python dissertation/presentation/make_demo_video.py

Writes dissertation/presentation/Demo_Video_Shpyl.mp4.
"""
from __future__ import annotations

import base64
import json
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_webapp_figures import Chrome, APP  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PRES = ROOT / "dissertation" / "presentation"
OUT = PRES / "Demo_Video_Shpyl.mp4"

W, H = 1280, 720          # 720p: large enough to read, small enough to mail
FPS = 24                  # output frame rate
CAPTURE_HZ = 8            # roughly, limited by how fast a screenshot round-trips
HOLD_STATIC = 2.6         # seconds a motionless stretch is allowed to occupy in the output
DIFF_THRESHOLD = 0.004    # fraction of changed pixels that counts as "something happened"

INK = (34, 40, 49)
PAPER = (250, 249, 246)
TEAL = (31, 78, 95)
CAPTION_H = 92

# A caption nobody can finish reading is worse than no caption, so each phase gets a floor on how
# long it stays on screen regardless of how little the page moved during it.
MIN_SECONDS = {
    "repo": 6, "open": 5, "loaded": 4.5, "s1": 8, "s2": 9, "s3": 9, "s4": 8,
    "inspect": 11, "cf": 8, "s5": 22, "contrast": 9, "guide": 10, "end": 6,
}


def font(size, bold=False):
    for name in (("calibrib.ttf", "arialbd.ttf") if bold else ("calibri.ttf", "arial.ttf")):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


# Each entry is the caption shown while that phase is on screen. They are written to be read, not
# heard, because the video has no narration and has to stand on its own in an email.
CAPTIONS = {
    "open": ("Served from 127.0.0.1",
             "Every model runs inside this process. Nothing a student wrote is uploaded."),
    "loaded": ("A machine-written essay, loaded",
               "Written by Llama 3.1 on this laptop for the detection corpus."),
    "s1": ("Step 1: the verdict",
           "0.96, flagged. Both readers shown separately: the fusion only reads high when they agree."),
    "s2": ("Step 2: why, in named habits",
           "Grey band is the middle 80% of 640 real student essays. The dot is this submission."),
    "s3": ("Step 3: how unusual, not just whether",
           "One extreme measurement means nothing. Several together are the pattern."),
    "s4": ("Step 4: which sentences the detector reacted to",
           "Each sentence deleted in turn and the essay re-scored. The marks are spread out."),
    "inspect": ("Click any marked sentence",
                "Rank, share of the signal, and what the sentence is not evidence of."),
    "cf": ("The verdict without its strongest evidence",
           "Remove the top sentences, re-score what is left, and see if the flag survives."),
    "s5": ("Step 5: the student's claims, and questions",
           "Sentence numbers kept, quotation looked up from the submission. It cannot be invented."),
    "contrast": ("The same pipeline on a real student's essay",
                 "0.0234, not flagged, and no questions generated. A tool that flags everything is useless."),
    "guide": ("What the lecturer actually receives",
              "The Verification Interview Guide: evidence for a conversation, not an accusation."),
    "end": ("Ninety-five seconds, one laptop, offline",
            "Code, results and dissertation: github.com/MichaelShpyl/AI_reserch"),
    "repo": ("Everything here is public",
             "The code, the results every number is read from, and the dissertation itself."),
}


def caption_frame(img: Image.Image, phase: str) -> Image.Image:
    """Letterbox the page and write the caption into the bar, so no page pixel is covered."""
    title, sub = CAPTIONS.get(phase, ("", ""))
    out = Image.new("RGB", (W, H + CAPTION_H), PAPER)
    out.paste(img.resize((W, H), Image.LANCZOS), (0, 0))
    d = ImageDraw.Draw(out)
    d.rectangle([0, H, W, H + 3], fill=TEAL)
    d.text((34, H + 16), title, font=font(25, bold=True), fill=INK)
    d.text((34, H + 52), sub, font=font(19), fill=(91, 103, 112))
    return out


def changed(a: Image.Image, b: Image.Image) -> float:
    """Fraction of the frame that moved, measured on a small greyscale copy."""
    sa = a.convert("L").resize((160, 90))
    sb = b.convert("L").resize((160, 90))
    pa, pb = sa.load(), sb.load()
    n = 0
    for y in range(0, 90, 2):
        for x in range(0, 160, 2):
            if abs(pa[x, y] - pb[x, y]) > 12:
                n += 1
    return n / (80 * 45)


def assemble(shots):
    """Turn the capture into an output timeline.

    Two rules. A stretch where nothing moves is compressed to a short hold, because watching a
    spinner for fifty seconds is not evidence of anything. And every phase gets a minimum time on
    screen, because its caption has to be readable at a normal reading speed."""
    per_phase: dict[str, list] = {}
    order: list[str] = []
    for _, phase, img in shots:
        if phase not in per_phase:
            per_phase[phase] = []
            order.append(phase)
        per_phase[phase].append(img)

    timeline: list[tuple[str, Image.Image]] = []
    max_static = int(HOLD_STATIC * CAPTURE_HZ)
    for phase in order:
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
        block = [(phase, img) for img in kept for _ in range(3)]   # 8 Hz capture -> 24 fps
        floor = int(MIN_SECONDS.get(phase, 5) * FPS)
        if len(block) < floor and kept:
            block += [(phase, kept[-1])] * (floor - len(block))
        timeline.extend(block)
    return timeline


def guide_frames():
    """The first pages of the generated guide, as still frames.

    The guide is the deliverable, so the video should end on it rather than on the tool that made
    it. Rendering the PDF directly is more honest than a screenshot of a PDF viewer, and it avoids
    depending on whether headless Chrome will display one."""
    pdf = ROOT / "outputs" / "verification_guides" / "3108a_ai_guide.pdf"
    if not pdf.exists():
        print("  no guide PDF, skipping that shot")
        return []
    import fitz
    doc = fitz.open(pdf)
    out = []
    for page in range(min(2, doc.page_count)):
        pix = doc[page].get_pixmap(dpi=150)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        # Fit the page into the frame without cropping, on the deck's paper colour.
        scale = min(W / img.width, H / img.height)
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
        canvas = Image.new("RGB", (W, H), PAPER)
        canvas.paste(img, ((W - img.width) // 2, (H - img.height) // 2))
        out.extend((0.0, "guide", canvas) for _ in range(int(CAPTURE_HZ * 5)))
    return out


def app_is_up(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(1.0)
        return s.connect_ex(("127.0.0.1", port)) == 0


def encode(frames: list[Image.Image], dest: Path):
    import imageio_ffmpeg
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    w, h = frames[0].size
    cmd = [exe, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(FPS), "-i", "-", "-an",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
           "-preset", "medium", "-movflags", "+faststart", str(dest)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    for f in frames:
        p.stdin.write(f.tobytes())
    p.stdin.close()
    if p.wait() != 0:
        sys.exit("ffmpeg failed")


def main():
    if not app_is_up(8000):
        sys.exit("The app is not running. Start it: python src/webapp/server.py")
    if not app_is_up(11434):
        print("WARNING: Ollama is not answering on 11434, so step 5 will show its error panel.\n"
              "         Start it with `ollama serve` and run this again.")

    profile = PRES / ".chrome_video"
    c = Chrome(profile)
    shots: list[tuple[float, str, Image.Image]] = []

    def grab(phase: str):
        data = c.send("Page.captureScreenshot", format="jpeg", quality=88)["data"]
        import io
        img = Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")
        shots.append((time.time(), phase, img))

    try:
        c.send("Emulation.setDeviceMetricsOverride", width=W, height=H,
               deviceScaleFactor=1, mobile=False)
        # Open on the repository, so the first thing a viewer sees is that this is not a mock-up.
        c.goto("https://github.com/MichaelShpyl/AI_reserch")
        c.wait_for("document.querySelectorAll('a').length > 20", timeout=60, label="github")
        c.js("""(() => { for (const sel of ['.js-notice','[data-testid=cookie-consent]','dialog'])
                 document.querySelectorAll(sel).forEach(e => e.remove()); })()""")
        for _ in range(14):
            grab("repo")

        c.goto(APP)
        c.wait_for("/ready/.test(document.querySelector('#pilltext').textContent)",
                   label="models to load")

        print("recording")
        for _ in range(10):
            grab("open")
        c.js("document.querySelector('#ex-ai').click()")
        c.wait_for("document.querySelector('#text').value.length > 500")
        for _ in range(12):
            grab("loaded")

        c.js("document.querySelector('#run').click()")

        # Follow the run: capture continuously, and scroll to whichever stage has just appeared.
        # The page keeps working while this loop runs, so the frames are the real progression.
        milestones = [
            ("s1", "document.querySelector('#s1 .body') && document.querySelector('#s1 .body').children.length"),
            ("s2", "document.querySelector('#s2 .body') && document.querySelector('#s2 .body').children.length"),
            ("s3", "document.querySelector('#s3 .body') && document.querySelector('#s3 .body').children.length"),
            ("s4", "document.querySelector('#s4 .body') && document.querySelector('#s4 .body').children.length"),
        ]
        phase = "s1"
        done = set()
        deadline = time.time() + 420
        while time.time() < deadline:
            for name, test in milestones:
                if name not in done and c.js(f"!!({test})"):
                    done.add(name)
                    phase = name
                    c.js(f"document.querySelector('#{name}')"
                         ".scrollIntoView({block:'start', behavior:'instant'})")
                    for _ in range(16):     # hold on each stage as it lands
                        grab(name)
            grab(phase)
            if len(done) == len(milestones):
                break

        # The evidence panel only exists after a click, and it is the most persuasive frame here.
        c.js("(document.querySelector('#doc3 s5.h3') || document.querySelector('#doc3 s5')).click()")
        c.wait_for("document.querySelector('#inspect .inspect')", timeout=60, label="evidence panel")
        c.js("document.querySelector('#inspect').scrollIntoView({block:'center', behavior:'instant'})")
        for _ in range(26):
            grab("inspect")

        c.wait_for("document.querySelector('#cfbox') && "
                   "!/Testing what happens/.test(document.querySelector('#cfbox').textContent)",
                   timeout=200, label="counterfactual")
        c.js("document.querySelector('#cfbox').scrollIntoView({block:'center', behavior:'instant'})")
        for _ in range(20):
            grab("cf")

        try:
            c.wait_for("document.querySelector('#s5 .body') && "
                       "document.querySelector('#s5 .body').querySelector('.claim, .err')",
                       timeout=420, label="claims and questions")
        except TimeoutError:
            print("  the question stage did not finish; recording what is on screen")
        c.js("document.querySelector('#s5').scrollIntoView({block:'start', behavior:'instant'})")
        for _ in range(18):
            grab("s5")
        # Scroll slowly through the claim cards so the questions are actually readable.
        for step in range(14):
            c.js(f"window.scrollBy(0, 150)")
            for _ in range(3):
                grab("s5")
        # The contrast. Only the verdict is needed, and that lands in about two seconds, so this
        # costs the video ten seconds rather than another ninety-five.
        c.js("window.scrollTo(0, 0)")
        # Wait for the box to actually CHANGE, not merely to be non-empty. It already holds the AI
        # essay at this point, so "length > 500" is true before the fetch has returned, and pressing
        # Analyse there re-runs the AI essay while the human text arrives underneath it. That is how
        # the first take ended up captioned "not flagged" over a verdict of 0.96.
        before = c.js("document.querySelector('#text').value.slice(0, 80)")
        c.js("document.querySelector('#ex-human').click()")
        c.wait_for("document.querySelector('#text').value.slice(0, 80) !== "
                   + json.dumps(before), timeout=30, label="the human example to load")
        c.js("document.querySelector('#run').click()")
        try:
            c.wait_for("document.querySelector('#s1 .body') && "
                       "document.querySelector('#s1 .body').children.length",
                       timeout=90, label="the contrast verdict")
            c.js("document.querySelector('#s1').scrollIntoView({block:'start', behavior:'instant'})")
            for _ in range(22):
                grab("contrast")
        except TimeoutError:
            print("  contrast run did not produce a verdict, skipping that shot")

        for _ in range(16):
            grab("end")
    finally:
        c.close()
        shutil.rmtree(profile, ignore_errors=True)

    print(f"captured {len(shots)} frames over {shots[-1][0] - shots[0][0]:.0f}s")

    # Cache the capture so the assembly below can be re-tuned without recording again.
    cache = PRES / ".demo_frames"
    if cache.exists():
        shutil.rmtree(cache, ignore_errors=True)
    cache.mkdir(parents=True, exist_ok=True)
    for i, (_, phase, img) in enumerate(shots):
        img.save(cache / f"{i:05d}_{phase}.jpg", quality=88)

    # The guide is a PDF rather than a page, so its frames come from the document itself.
    shots.extend(guide_frames())

    timeline = assemble(shots)

    print(f"timeline {len(timeline)} frames = {len(timeline) / FPS:.0f}s at {FPS} fps")
    frames = [caption_frame(img, phase) for phase, img in timeline]
    encode(frames, OUT)
    size_mb = OUT.stat().st_size / 1e6
    print(f"wrote {OUT}  {len(frames) / FPS:.0f}s  {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
