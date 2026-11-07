"""Motion graphics and audio helpers for the promo film. Everything is drawn here and rendered locally."""

import json
import math
import random
import subprocess
from collections.abc import Iterator
from functools import cache
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, FPS, SR = 1920, 1080, 30, 44100
FF = imageio_ffmpeg.get_ffmpeg_exe()
FONTS = Path("C:/Windows/Fonts")
ROOT = Path(__file__).resolve().parent.parent
BLACK, WHITE, INK = (6, 12, 14), (255, 255, 255), (18, 52, 59)
GREY_D, PAPER = (134, 150, 155), (242, 248, 247)
TEAL, TEAL_D = (94, 214, 190), (15, 118, 110)
GREEN, RED = (27, 127, 75), (179, 38, 30)
REC_W, REC_H = 1600, 1000


@cache
def font(weight: str, size: int) -> ImageFont.FreeTypeFont:
    name = {
        "bold": "segoeuib.ttf",
        "semi": "seguisb.ttf",
        "reg": "segoeui.ttf",
        "light": "segoeuil.ttf",
        "serif": "georgia.ttf",
        "serifb": "georgiab.ttf",
    }[weight]
    return ImageFont.truetype(str(FONTS / name), size)


def ease(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return 1 - (1 - x) ** 4


def ease_io(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def env(t: float, start: float, dur: float, fin: float = 0.9, fout: float = 0.6) -> tuple[float, float]:
    """(alpha, rise) for a fade-up entrance at `start` and a fade-out at the end of `dur`."""
    k = ease((t - start) / fin)
    return min(k, ease((dur - t) / fout)), 18 * (1 - k)


def put(im, s, f, cy, color, alpha, rise=0.0, cx=W / 2) -> None:
    if alpha <= 0.003 or not s:
        return
    box = f.getbbox(s)
    x = cx - (box[2] - box[0]) / 2 - box[0]
    y = cy - (box[3] - box[1]) / 2 - box[1] + rise
    mask = Image.new("L", im.size, 0)
    ImageDraw.Draw(mask).text((x, y), s, font=f, fill=int(255 * alpha))
    im.paste(Image.new("RGB", im.size, color), (0, 0), mask)


def words(im, s, f, cy, color, t, start, dur, gap=0.14) -> None:
    """Kinetic type: each word rises into place in turn."""
    parts = s.split(" ")
    widths = [f.getlength(p + " ") for p in parts]
    x = W / 2 - sum(widths) / 2
    for i, (p, w) in enumerate(zip(parts, widths, strict=True)):
        a, r = env(t, start + i * gap, dur, 0.7, 0.6)
        put(im, p, f, cy, color, a, r * 1.6, cx=x + f.getlength(p) / 2)
        x += w


@cache
def text_wall() -> Image.Image:
    """A tall sheet of the project's own sample forwards, set small and grey."""
    messages = json.loads((ROOT / "data" / "messages.json").read_text(encoding="utf-8"))
    f = font("reg", 24)
    im = Image.new("L", (W + 400, H * 3), 0)
    d = ImageDraw.Draw(im)
    rnd = random.Random(3)
    y, col_w, x = 0, 560, 30
    all_words = " ".join(m["text"] for m in messages * 40).split()
    line = ""
    for word in all_words:
        if f.getlength(line + word) > col_w:
            d.text((x, y), line, font=f, fill=rnd.randint(70, 150))
            line, y = "", y + 38
            if y > im.height - 40:
                y, x = rnd.randint(0, 30), x + col_w + 60
                if x > im.width - col_w:
                    break
        line += word + " "
    return im


@cache
def vignette() -> np.ndarray:
    yy, xx = np.mgrid[-1 : 1 : complex(0, H), -1 : 1 : complex(0, W)]
    return np.clip(1.15 - (xx**2 * 0.8 + yy**2), 0, 1).astype(np.float32)


def wall(t, speed=26, strength=0.26, blur=1.6) -> Image.Image:
    """Endless drifting forwards, out of focus, fading at the edges."""
    sheet = text_wall()
    y = int(t * speed) % (sheet.height - H)
    x = int(120 + 60 * math.sin(t * 0.15))
    crop = sheet.crop((x, y, x + W, y + H)).filter(ImageFilter.GaussianBlur(blur))
    arr = np.asarray(crop, dtype=np.float32) * strength * vignette()
    base = np.array(BLACK, dtype=np.float32)
    return Image.fromarray((np.dstack([arr] * 3) + base).clip(0, 255).astype("uint8"))


def bubble(im, x, y, text, t, start, dur, width=760, forwarded=True) -> None:
    """A chat bubble popping in, with the 'Forwarded' tag messaging apps add."""
    k = ease((t - start) / 0.45)
    a = min(k, ease((dur - t) / 0.5))
    if a <= 0.01:
        return
    f, small = font("reg", 34), font("reg", 24)
    lines, line = [], ""
    for w in text.split():
        if f.getlength(line + w) > width - 70:
            lines.append(line)
            line = ""
        line += w + " "
    lines.append(line)
    h = 92 + 46 * len(lines)
    y = y + 28 * (1 - k)
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.rounded_rectangle((x, y, x + width, y + h), 22, fill=(32, 58, 54, int(245 * a)))
    if forwarded:
        d.text((x + 32, y + 18), "Forwarded many times", font=small, fill=(150, 180, 175, int(255 * a)))
    for i, s in enumerate(lines):
        d.text((x + 32, y + 56 + 46 * i), s, font=f, fill=(235, 245, 243, int(255 * a)))
    im.paste(layer, (0, 0), layer)


def bloom(im, k) -> Image.Image:
    """Light opening up from the centre: the turn from doubt to clarity."""
    if k <= 0:
        return im
    yy, xx = np.mgrid[-1 : 1 : complex(0, H // 4), -1.78 : 1.78 : complex(0, W // 4)]
    r = np.sqrt(xx**2 + yy**2)
    glow = np.exp(-((r / (0.25 + 1.4 * k)) ** 2)) * k
    tint = np.array((94, 214, 190), dtype=np.float32)
    layer = Image.fromarray((glow[..., None] * tint).clip(0, 255).astype("uint8")).resize((W, H), Image.BILINEAR)
    return Image.fromarray(np.minimum(255, np.asarray(im, np.int16) + np.asarray(layer, np.int16)).astype("uint8"))


PARTICLES = [(random.Random(i).random(), random.Random(i * 7).random(), random.Random(i * 13).uniform(0.4, 1.4)) for i in range(90)]


def motes(im, t, k) -> None:
    """Small lights drifting upward."""
    if k <= 0:
        return
    d = ImageDraw.Draw(im, "RGBA")
    for px, py, sp in PARTICLES:
        y = (py * H - t * 40 * sp) % H
        x = px * W + 18 * math.sin(t * sp + px * 9)
        r = 1.5 + 2.2 * sp
        d.ellipse((x - r, y - r, x + r, y + r), fill=(170, 240, 225, int(150 * k * (0.4 + 0.6 * sp))))


def lens(im, cx, cy, size, progress, alpha=1.0, color=TEAL) -> None:
    """A magnifier that draws itself, then a check mark inside it, with a soft glow."""
    if progress <= 0 or alpha <= 0:
        return
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    rgba = (*color, int(255 * alpha))
    r, w = size * 0.36, max(4, int(size / 16))
    ox, oy = cx - size * 0.1, cy - size * 0.1
    d.arc((ox - r, oy - r, ox + r, oy + r), -90, -90 + 360 * min(1, progress * 2), fill=rgba, width=w)
    if progress > 0.5:
        k = min(1, (progress - 0.5) * 4)
        a0 = ox + r * math.cos(math.pi / 4)
        b0 = oy + r * math.sin(math.pi / 4)
        d.line((a0, b0, a0 + size * 0.34 * k, b0 + size * 0.34 * k), fill=rgba, width=int(w * 1.5))
    if progress > 0.72:
        k = min(1, (progress - 0.72) / 0.28)
        pts = [(ox - r * 0.5, oy), (ox - r * 0.12, oy + r * 0.4), (ox + r * 0.55, oy - r * 0.42)]
        first = min(1, k * 2)
        path = [pts[0], (pts[0][0] + (pts[1][0] - pts[0][0]) * first, pts[0][1] + (pts[1][1] - pts[0][1]) * first)]
        if k > 0.5:
            second = (k - 0.5) * 2
            path.append((pts[1][0] + (pts[2][0] - pts[1][0]) * second, pts[1][1] + (pts[2][1] - pts[1][1]) * second))
        d.line(path, fill=rgba, width=w, joint="curve")
    glow = layer.filter(ImageFilter.GaussianBlur(10))
    im.paste(glow, (0, 0), glow)
    im.paste(layer, (0, 0), layer)


def verdict_chip(im, label, color, cx, cy, t, start, dur) -> None:
    """A verdict label that pops in."""
    k = ease((t - start) / 0.45)
    a = min(k, ease((dur - t) / 0.4))
    if a <= 0.01:
        return
    f = font("semi", 46)
    width = f.getlength(label) + 80
    x0, y0 = cx - width / 2, cy - 42 + 20 * (1 - k)
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    ImageDraw.Draw(layer).rounded_rectangle((x0, y0, x0 + width, y0 + 84), 42, fill=(*color, int(255 * a)))
    ImageDraw.Draw(layer).text((x0 + 40, y0 + 12), label, font=f, fill=(255, 255, 255, int(255 * a)))
    im.paste(layer, (0, 0), layer)


# --- the real app, framed and zoomed ---------------------------------------------------------

WIN_W, WIN_H, BAR = 1520, 950, 40


@cache
def window_assets():
    m = 90
    shadow = Image.new("RGBA", (WIN_W + 2 * m, WIN_H + BAR + 2 * m), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle((m, m + 28, m + WIN_W, m + WIN_H + BAR + 28), 24, fill=(0, 0, 0, 55))
    shadow = shadow.filter(ImageFilter.GaussianBlur(42))
    bar = Image.new("RGB", (WIN_W, BAR), (236, 243, 242))
    d = ImageDraw.Draw(bar)
    for i in range(3):
        d.ellipse((20 + i * 22, 14, 31 + i * 22, 25), fill=(200, 212, 210))
    mask = Image.new("L", (WIN_W, WIN_H + BAR), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, WIN_W - 1, WIN_H + BAR - 1), 24, fill=255)
    return shadow, bar, mask


def focus_rect(box, k):
    """Crop of the recording, easing from the full frame to the box of interest."""
    full = (0.0, 0.0, float(REC_W), float(REC_H))
    if not box or k <= 0:
        return full
    x, y, w, h = box
    pad = 70
    tw = max(860.0, w + 2 * pad)
    th = tw * REC_H / REC_W
    if h + 2 * pad > th:
        th = h + 2 * pad
        tw = th * REC_W / REC_H
    tw, th = min(tw, REC_W), min(th, REC_H)
    tx = min(max(0.0, x + w / 2 - tw / 2), REC_W - tw)
    ty = min(max(0.0, y + h / 2 - th / 2), REC_H - th)
    return tuple(a + (b - a) * k for a, b in zip(full, (tx, ty, tx + tw, ty + th), strict=True))


def framed(frame, t, dur, caption, box, zoom_k, glow_k) -> Image.Image:
    """The app in a clean window, pushing in to the part that matters, with a highlight around it."""
    bg = Image.new("RGB", (W, H), PAPER)
    shadow, bar, mask = window_assets()
    rect = focus_rect(box, zoom_k)
    view = frame.crop(tuple(int(v) for v in rect)).resize((WIN_W, WIN_H), Image.BILINEAR)
    if box and glow_k > 0:
        sx, sy = WIN_W / (rect[2] - rect[0]), WIN_H / (rect[3] - rect[1])
        x, y, w, h = box
        bx0, by0 = (x - rect[0]) * sx - 10, (y - rect[1]) * sy - 10
        bx1, by1 = bx0 + w * sx + 20, by0 + h * sy + 20
        hl = Image.new("RGBA", view.size, (0, 0, 0, 0))
        pulse = 0.75 + 0.25 * math.sin(t * 3.2)
        ImageDraw.Draw(hl).rounded_rectangle((bx0, by0, bx1, by1), 18, outline=(*TEAL_D, int(235 * glow_k * pulse)), width=4)
        g = hl.filter(ImageFilter.GaussianBlur(8))
        view.paste(g, (0, 0), g)
        view.paste(hl, (0, 0), hl)
    win = Image.new("RGB", (WIN_W, WIN_H + BAR), WHITE)
    win.paste(bar, (0, 0))
    win.paste(view, (0, BAR))
    s = 0.95 + 0.025 * ease_io(t / dur)  # slow push-in of the whole window
    ww, wh = int(WIN_W * s), int((WIN_H + BAR) * s)
    win_z, mask_z = win.resize((ww, wh), Image.BILINEAR), mask.resize((ww, wh), Image.BILINEAR)
    x, y = (W - ww) // 2, 120 + (H - 120 - wh) // 2
    sh = shadow.resize((int(shadow.width * s), int(shadow.height * s)))
    bg.paste(sh, (x - int(90 * s), y - int(90 * s)), sh)
    bg.paste(win_z, (x, y), mask_z)
    a, r = env(t, 0.1, dur, 0.7, 0.5)
    put(bg, caption, font("semi", 42), 62, INK, a, r * 0.5)
    return bg


def screen_frames(rec: Path, start, end, n_out) -> Iterator[tuple[Image.Image, float]]:
    """n_out frames spanning the recording from start to end, re-timed to fit; yields (frame,
    source time) so zooms can follow what the recording is showing."""
    span = max(0.1, end - start)
    proc = subprocess.Popen(
        [FF, "-loglevel", "error", "-ss", f"{start:.2f}", "-t", f"{span:.2f}", "-i", str(rec),
         "-vf", f"fps={FPS},scale={REC_W}:{REC_H}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE,
    )
    size = REC_W * REC_H * 3
    n_src = max(1, int(span * FPS))
    last, idx = None, -1
    for k in range(n_out):
        want = min(n_src - 1, int(k * n_src / n_out))
        while idx < want:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            last = Image.frombytes("RGB", (REC_W, REC_H), buf)
            idx += 1
        yield (last if last is not None else Image.new("RGB", (REC_W, REC_H), WHITE)), start + want / FPS
    proc.stdout.close()
    proc.wait()


# --- audio -----------------------------------------------------------------------------------

TRACK_REVEAL = 13.0  # seconds into "The Product Reveal" where it opens up


def music_bed(track: Path, total, turn) -> np.ndarray:
    """The soundtrack, shifted so its reveal lands exactly on the picture's turn."""
    raw = subprocess.run(
        [FF, "-loglevel", "error", "-i", str(track), "-ac", "2", "-ar", str(SR), "-f", "f32le", "-"],
        capture_output=True, check=True,
    ).stdout
    samples = np.frombuffer(raw, dtype="<f4").reshape(-1, 2).astype(np.float64)
    n = int(total * SR)
    offset = int((turn - TRACK_REVEAL) * SR)
    out = np.zeros((n, 2))
    if offset >= 0:
        seg = samples[: n - offset]
        out[offset : offset + len(seg)] = seg
    else:
        seg = samples[-offset:][:n]
        out[: len(seg)] = seg
    return out


def smooth(x, win):
    """Centred moving average in O(n)."""
    c = np.concatenate([[0.0], np.cumsum(x, dtype=np.float64)])
    half = win // 2
    idx = np.arange(len(x))
    lo, hi = np.clip(idx - half, 0, len(x)), np.clip(idx + half + 1, 0, len(x))
    return (c[hi] - c[lo]) / (hi - lo)


def finish(sig, peak=0.89):
    """Fade in and out, and normalise."""
    n = len(sig)
    fade = np.clip(np.arange(n) / (0.5 * SR), 0, 1) * np.clip((n - np.arange(n)) / (3.0 * SR), 0, 1)
    sig = sig * fade[:, None]
    return sig / max(1e-6, np.abs(sig).max()) * peak


def write_wav(path: Path, sig) -> None:
    import wave

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes((np.clip(sig, -1, 1) * 32767).astype("<i2").tobytes())


def encoder(path: Path) -> subprocess.Popen:
    return subprocess.Popen(
        [FF, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
         "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p", str(path)],
        stdin=subprocess.PIPE,
    )


def mux(video: Path, wav: Path, out: Path) -> None:
    subprocess.run(
        [FF, "-y", "-loglevel", "error", "-i", str(video), "-i", str(wav), "-c:v", "copy", "-c:a", "aac",
         "-b:a", "256k", "-shortest", "-movflags", "+faststart", str(out)],
        check=True,
    )
