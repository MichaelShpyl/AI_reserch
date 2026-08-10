"""Screenshot the running web app for the implementation chapter.

The figures in Section 4.11 have to show the real interface, not a mock-up, so this drives a
headless Chrome over the DevTools protocol: load the page, click through the four stages, wait for
each one to settle, then capture it. Chrome is scripted rather than screenshotted by hand because a
hand-taken screenshot goes stale the moment the layout changes and nobody notices.

Start the app first (python src/webapp/server.py), then:

    python dissertation/presentation/make_webapp_figures.py

Writes fig_webapp_*.png into dissertation/figures/.
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

import requests
import websocket  # websocket-client

ROOT = Path(__file__).resolve().parents[2]
FIGS = ROOT / "dissertation" / "figures"
APP = "http://127.0.0.1:8000"
PORT = 9222
WIDTH = 1180          # wide enough for the two-column compare view
SCALE = 2             # retina, so the text is still sharp when Word scales it down

CHROME_CANDIDATES = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "/usr/bin/google-chrome",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
]


def find_chrome() -> str:
    for c in CHROME_CANDIDATES:
        if Path(c).exists():
            return c
    found = shutil.which("chrome") or shutil.which("google-chrome") or shutil.which("chromium")
    if found:
        return found
    sys.exit("No Chrome binary found. Edit CHROME_CANDIDATES.")


class Chrome:
    """The smallest CDP client that does the job: evaluate, wait, capture."""

    def __init__(self, profile: Path):
        self.proc = subprocess.Popen([
            find_chrome(),
            "--headless=new",
            f"--remote-debugging-port={PORT}",
            f"--user-data-dir={profile}",
            f"--window-size={WIDTH},2400",
            "--hide-scrollbars",
            "--force-device-scale-factor=%d" % SCALE,
            # Chrome refuses cross-origin DevTools sockets by default, and the client here counts
            # as a different origin from the debugging port itself.
            "--remote-allow-origins=*",
            "--no-first-run", "--no-default-browser-check", "--disable-gpu",
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        target = self._wait_for_target()
        try:
            self.ws = websocket.create_connection(target, timeout=60)
        except Exception:
            # Usually means a Chrome left over from an earlier run still owns the port. Clean up
            # this one rather than leaving a second orphan behind.
            self.proc.terminate()
            raise
        self.msg_id = 0
        self.send("Page.enable")
        self.send("Runtime.enable")
        # The page follows the operating system theme. Force the light one: these figures end up
        # printed on white paper, where a dark screenshot is a solid block of ink.
        self.send("Emulation.setEmulatedMedia",
                  features=[{"name": "prefers-color-scheme", "value": "light"}])
        # Pin the viewport rather than relying on the window size, which the device scale factor
        # rescales behind your back. A fixed width is also what makes the figures reproducible.
        self.send("Emulation.setDeviceMetricsOverride",
                  width=WIDTH, height=900, deviceScaleFactor=SCALE, mobile=False)

    def _wait_for_target(self) -> str:
        for _ in range(120):
            try:
                pages = [t for t in requests.get(f"http://127.0.0.1:{PORT}/json", timeout=2).json()
                         if t["type"] == "page"]
                if pages:
                    return pages[0]["webSocketDebuggerUrl"]
            except Exception:
                pass
            time.sleep(0.25)
        sys.exit("Chrome never opened a debugging port.")

    def send(self, method: str, **params):
        self.msg_id += 1
        self.ws.send(json.dumps({"id": self.msg_id, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self.msg_id:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    def js(self, expr: str):
        r = self.send("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
        if r.get("exceptionDetails"):
            raise RuntimeError(r["exceptionDetails"].get("text", "js error"))
        return r.get("result", {}).get("value")

    def goto(self, url: str):
        self.send("Page.navigate", url=url)
        self.wait_for("document.readyState === 'complete'")

    def wait_for(self, expr: str, timeout: float = 240.0, label: str = ""):
        """Poll a JavaScript predicate. Detection takes seconds and questions take a minute or
        more, so everything is waited for rather than slept through."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                if self.js(f"!!({expr})"):
                    return
            except RuntimeError:
                pass
            time.sleep(0.4)
        raise TimeoutError(f"timed out waiting for {label or expr}")

    def shot(self, name: str, clip_sel: str | None = None):
        """Capture the whole page, or one element if a selector is given."""
        params = {"format": "png", "captureBeyondViewport": True}
        if clip_sel:
            box = self.js(
                "(() => { const e = document.querySelector(%s); if (!e) return null;"
                " const r = e.getBoundingClientRect();"
                " return {x: r.x + scrollX, y: r.y + scrollY, w: r.width, h: r.height}; })()"
                % json.dumps(clip_sel))
            if not box:
                raise RuntimeError(f"selector not found: {clip_sel}")
            params["clip"] = {"x": box["x"], "y": box["y"], "width": box["w"],
                              "height": box["h"], "scale": SCALE}
        else:
            # Down to the bottom of the last thing on the page, not the bottom of the scroll area.
            # The two differ by whatever empty space the viewport leaves, which on a short page is
            # most of the figure.
            m = self.js(
                "(() => { const els = [...document.body.querySelectorAll('*')];"
                " const bottom = Math.max(...els.map(e => e.getBoundingClientRect().bottom + scrollY));"
                " return {w: document.documentElement.clientWidth, h: Math.ceil(bottom) + 16}; })()")
            params["clip"] = {"x": 0, "y": 0, "width": m["w"], "height": m["h"], "scale": SCALE}
        data = self.send("Page.captureScreenshot", **params)["data"]
        out = FIGS / f"{name}.png"
        out.write_bytes(base64.b64decode(data))
        print(f"  wrote {out.name}  ({out.stat().st_size // 1024} KB)")

    def close(self):
        try:
            self.ws.close()
        finally:
            self.proc.terminate()


def contact_sheet():
    """One figure showing the four analysis stages at once, for the presentation.

    The dissertation gets a figure per stage, because a reader can turn pages. A slide cannot, so
    the deck needs the whole shape of the interface in a single image."""
    from PIL import Image, ImageDraw, ImageFont

    tiles = [("fig_webapp_verdict", "1. Is this likely AI-written?"),
             ("fig_webapp_habits", "2. Why, in named writing habits"),
             ("fig_webapp_percentiles", "3. How unusual, against 640 real essays"),
             ("fig_webapp_questions", "4. Claims, and questions to ask")]
    PAPER, TEAL, LINE = (250, 249, 246), (31, 78, 95), (214, 214, 208)
    CELL_W, PAD, LABEL_H, GAP = 1500, 30, 52, 26

    def fnt(size):
        for n in ("calibrib.ttf", "calibri.ttf"):
            try:
                return ImageFont.truetype(n, size)
            except OSError:
                pass
        return ImageFont.load_default()

    cells = []
    for name, label in tiles:
        im = Image.open(FIGS / f"{name}.png").convert("RGB")
        # Tall screenshots are cropped from the top rather than squashed, so the text stays legible.
        im = im.resize((CELL_W, round(im.height * CELL_W / im.width)), Image.LANCZOS)
        im = im.crop((0, 0, CELL_W, min(im.height, round(CELL_W * 0.45))))
        cell = Image.new("RGB", (CELL_W, im.height + LABEL_H), PAPER)
        d = ImageDraw.Draw(cell)
        d.text((2, 4), label.upper(), font=fnt(30), fill=TEAL)
        cell.paste(im, (0, LABEL_H))
        d.rectangle([0, LABEL_H, CELL_W - 1, LABEL_H + im.height - 1], outline=LINE, width=3)
        cells.append(cell)

    rh = [max(cells[0].height, cells[1].height), max(cells[2].height, cells[3].height)]
    W = PAD * 2 + CELL_W * 2 + GAP
    H = PAD * 2 + rh[0] + rh[1] + GAP
    out = Image.new("RGB", (W, H), PAPER)
    for i, c in enumerate(cells):
        x = PAD + (i % 2) * (CELL_W + GAP)
        y = PAD + (0 if i < 2 else rh[0] + GAP)
        out.paste(c, (x, y))
    dest = FIGS / "fig_webapp_stages.png"
    out.save(dest, optimize=True)
    print(f"  wrote {dest.name}  {out.width}x{out.height}  ratio {out.height / out.width:.2f}")


def app_is_up() -> bool:
    with socket.socket() as s:
        s.settimeout(1.0)
        return s.connect_ex(("127.0.0.1", 8000)) == 0


def main():
    if not app_is_up():
        sys.exit("The app is not running. Start it with: python src/webapp/server.py")
    FIGS.mkdir(parents=True, exist_ok=True)
    profile = Path(__file__).resolve().parent / ".chrome_profile"
    c = Chrome(profile)
    try:
        print("landing page")
        c.goto(APP)
        c.wait_for("/ready/.test(document.querySelector('#pilltext').textContent)",
                   label="models to finish loading")
        c.shot("fig_webapp_landing")

        print("running the AI-written example through all five steps")
        c.js("document.querySelector('#ex-ai').click()")
        c.wait_for("document.querySelector('#text').value.length > 500", label="example text")
        c.js("document.querySelector('#run').click()")

        c.wait_for("document.querySelector('#s1 .body') && "
                   "document.querySelector('#s1 .body').children.length", label="verdict")
        c.shot("fig_webapp_verdict", "#s1")

        c.wait_for("document.querySelector('#s2 .body') && "
                   "document.querySelector('#s2 .body').children.length", label="habit card")
        c.shot("fig_webapp_habits", "#s2")

        c.wait_for("document.querySelector('#s3 .body') && "
                   "document.querySelector('#s3 .body').children.length", label="percentiles")
        c.shot("fig_webapp_percentiles", "#s3")

        c.wait_for("document.querySelector('#s4 .body') && "
                   "document.querySelector('#s4 .body').children.length", label="sentence marks")
        # Step 4 also re-scores the essay with its strongest sentences removed, which takes a few
        # seconds. Capturing before that lands would show a spinner where the result belongs.
        c.wait_for("document.querySelector('#cfbox') && "
                   "!/Testing what happens/.test(document.querySelector('#cfbox').textContent)",
                   timeout=180, label="the counterfactual")
        c.shot("fig_webapp_sentences", "#s4")

        # Clicking a marked sentence opens the evidence panel, which is the part of the interface
        # that answers "so what do I do with this". It only exists after a click, so click one.
        # h3 is the darkest shading, so it is the sentence the detector reacted to most.
        c.js("(document.querySelector('#doc3 s5.h3') || document.querySelector('#doc3 s5')).click()")
        c.wait_for("document.querySelector('#inspect .inspect')", timeout=60,
                   label="the evidence panel")
        c.shot("fig_webapp_inspect", "#inspect .inspect")

        # The question stage runs a language model, so it is the slow one and it is allowed to fail
        # (Ollama may not be running). A missing stage 5 must not cost us the four figures above.
        try:
            c.wait_for("document.querySelector('#s5 .body') && "
                       "document.querySelector('#s5 .body').querySelector('.claim, .err')",
                       timeout=420, label="claims and questions")
            # Four claim cards make a figure two pages long. One card shows the whole structure:
            # the claim, the sentence numbers it came from, the quoted source, the questions and
            # their Bloom levels. The rest is the same shape repeated.
            c.shot("fig_webapp_questions", "#s5 .claim")
        except TimeoutError as e:
            print("  skipped the question stage:", e)

    finally:
        c.close()
        shutil.rmtree(profile, ignore_errors=True)


if __name__ == "__main__":
    main()
