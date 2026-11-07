"""Render the two-minute demo film: the problem (0-15 s, in the dark), the turn at 15 s, then the real
app, with zooms and highlights, narrated by a local text-to-speech voice.

uv run --no-project --python 3.12 --with pillow --with numpy --with imageio-ffmpeg --with kokoro-onnx \
    python promo/film.py REC_DIR KOKORO_DIR TRACK.mp3 OUT.mp4 [WORK_DIR] [--stills]
KOKORO_DIR holds kokoro-v1.0.onnx and voices-v1.0.bin. REC_DIR comes from record.py.
--stills writes one frame from the middle of each scene to WORK_DIR and stops.
"""

import json
import sys
import wave
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from motion import (
    BLACK, FPS, GREEN, GREY_D, H, INK, PAPER, RED, SR, TEAL, TEAL_D, W, WHITE,
    bloom, bubble, ease, encoder, env, finish, font, framed, lens, motes, music_bed, mux, put,
    screen_frames, smooth, verdict_chip, wall, words, write_wav,
)
from PIL import Image, ImageDraw

TURN = 15.0
GRAY = (107, 111, 118)


@dataclass
class Scene:
    vo: str
    render: Callable[[float, float], Image.Image] | None = None
    ui: tuple[str, str, str | None, float] | None = None  # (start mark, end mark, focus mark, zoom lead)
    caption: str = ""
    min_dur: float = 3.0
    max_speedup: float = 1.9  # how much faster than real time the recording may play


def s_forward(t, d):
    im = wall(t)
    bubble(im, 150, 170, "URGENT!! Share with everyone you know before they delete this!!", t, 0.2, d)
    bubble(im, 1010, 390, "Doctors don't want you to know this one simple cure.", t, 1.0, d)
    bubble(im, 300, 640, "Government is giving free money to all who forward this message.", t, 1.8, d)
    return im


def s_sharing(t, d):
    im = wall(t + 4, speed=46, strength=0.30)
    words(im, "It sounds true.", font("serif", 92), H / 2 - 70, WHITE, t, 0.2, d)
    words(im, "Everyone is sharing it.", font("serif", 72), H / 2 + 60, GREY_D, t, 1.0, d)
    return im


def s_question(t, d):
    im = Image.new("RGB", (W, H), BLACK)
    words(im, "But who actually checked?", font("serifb", 96), H / 2, WHITE, t, 0.25, d)
    return im


def s_title(t, d):
    im = bloom(Image.new("RGB", (W, H), BLACK), 0.42 * ease(t / 2.2))
    motes(im, t, ease(t / 2.5) * env(t, 0, d, 0.1, 0.7)[0])
    lens(im, W / 2, H / 2 - 170, 240, ease(t / 1.8), env(t, 0, d, 0.1, 0.6)[0])
    a, r = env(t, 1.3, d, 0.9, 0.6)
    put(im, "Claim Checker", font("serifb", 140), H / 2 + 70, WHITE, a, r)
    a, r = env(t, 2.2, d, 0.9, 0.6)
    put(im, "For viral messages.", font("reg", 46), H / 2 + 190, TEAL, a, r)
    return im


def s_verdicts(t, d):
    im = Image.new("RGB", (W, H), PAPER)
    a, r = env(t, 0.1, d, 0.6, 0.4)
    put(im, "Every claim gets a verdict.", font("serif", 74), H / 2 - 150, INK, a, r)
    verdict_chip(im, "Supported", GREEN, W / 2 - 400, H / 2 + 40, t, 0.6, d)
    verdict_chip(im, "Refuted", RED, W / 2, H / 2 + 40, t, 1.3, d)
    verdict_chip(im, "Unverified", GRAY, W / 2 + 400, H / 2 + 40, t, 2.0, d)
    return im


STAGES = [("Read", "OCR, then redaction"), ("Split", "Language model"), ("Search", "Wikipedia and news"), ("Judge", "DeBERTa classifier")]


def s_pipeline(t, d):
    """The four stages appear in turn, joined by arrows that draw themselves."""
    im = Image.new("RGB", (W, H), PAPER)
    a, r = env(t, 0.1, d, 0.6, 0.4)
    put(im, "How it works", font("serif", 72), 230, INK, a, r)
    draw = ImageDraw.Draw(im)
    bw, bh, gap = 330, 190, 90
    x0 = (W - (4 * bw + 3 * gap)) / 2
    step = max(0.9, (d - 2.5) / 4)
    for i, (name, about) in enumerate(STAGES):
        start = 0.5 + i * step
        k = ease((t - start) / 0.5)
        alpha = min(k, ease((d - t) / 0.5))
        if alpha <= 0.01:
            continue
        x, y = x0 + i * (bw + gap), H / 2 - bh / 2 + 24 * (1 - k)
        layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        ld.rounded_rectangle((x, y, x + bw, y + bh), 26, fill=(255, 255, 255, int(255 * alpha)),
                             outline=(*TEAL_D, int(255 * alpha)), width=4)
        im.paste(layer, (0, 0), layer)
        put(im, name, font("semi", 52), y + 72, INK, alpha, 0, cx=x + bw / 2)
        put(im, about, font("reg", 28), y + 132, (82, 96, 109), alpha, 0, cx=x + bw / 2)
        if i:
            p = ease((t - start + 0.35) / 0.5)
            ax0, ay = x - gap + 12, H / 2
            ax1 = ax0 + (gap - 24) * p
            draw.line((ax0, ay, ax1, ay), fill=TEAL_D, width=5)
            if p > 0.95:
                draw.polygon([(ax1 + 4, ay), (ax1 - 14, ay - 11), (ax1 - 14, ay + 11)], fill=TEAL_D)
    a, r = env(t, 0.5 + 4 * step, d, 0.6, 0.4)
    put(im, "The language model never decides the verdict.", font("reg", 40), H / 2 + 230, TEAL_D, a, r)
    return im


def s_privacy(t, d):
    im = Image.new("RGB", (W, H), PAPER)
    a, r = env(t, 0.1, d, 0.6, 0.4)
    put(im, "Private details stay out.", font("serif", 72), 250, INK, a, r)
    split = d * 0.45
    if t < split:
        bubble(im, 480, 420, "Call Ravi on 98765 43210 or mail ravi@example.com for the cure.", t, 0.4, split, width=960)
    else:
        bubble(im, 480, 420, "Call Ravi on [phone] or mail [email] for the cure.", t, split, d, width=960)
    a, r = env(t, split + 0.4, d, 0.6, 0.4)
    put(im, "Removed before the message reaches the AI.", font("reg", 40), 720, TEAL_D, a, r)
    return im


def s_score(t, d):
    """An honest number: the held-out result, counting up, and where the mistakes are written down."""
    im = Image.new("RGB", (W, H), WHITE)
    n = min(10, int(11 * ease((t - 0.4) / 1.6)))
    a, r = env(t, 0.2, d, 0.5, 0.4)
    put(im, f"{n} / 12", font("serifb", 200), H / 2 - 110, INK, a, r)
    a, r = env(t, 0.8, d, 0.6, 0.4)
    put(im, "held-out claims right, on claims it was never tuned on", font("reg", 44), H / 2 + 60, (82, 96, 109), a, r)
    a, r = env(t, 2.6, d, 0.6, 0.4)
    put(im, "Every failure case is documented.", font("semi", 50), H / 2 + 190, TEAL_D, a, r)
    return im


PILLARS = [
    ("Every claim, checked.", "Against Wikipedia and trusted news."),
    ("Every source, linked.", "Read the evidence yourself."),
]


def s_pillars(t, d):
    seg = d / len(PILLARS)
    i = min(len(PILLARS) - 1, int(t // seg))
    lt = t - i * seg
    head, sub = PILLARS[i]
    im = Image.new("RGB", (W, H), WHITE)
    lens(im, W / 2, H / 2 - 190, 170, ease(lt / 0.8), 1.0, TEAL_D)
    a, r = env(lt, 0.2, seg, 0.55, 0.4)
    put(im, head, font("serifb", 104), H / 2 + 30, INK, a, r)
    a, r = env(lt, 0.5, seg, 0.6, 0.4)
    put(im, sub, font("reg", 44), H / 2 + 150, (82, 96, 109), a, r)
    return im


def s_end(t, d):
    im = bloom(Image.new("RGB", (W, H), BLACK), 0.14 * env(t, 0, d, 1.5, 1.5)[0])
    lens(im, W / 2, H / 2 - 215, 150, ease(t / 1.4), env(t, 0, d, 0.2, 1.2)[0])
    a, r = env(t, 0.6, d, 1.0, 1.2)
    put(im, "Claim Checker", font("serifb", 130), H / 2 - 10, WHITE, a, r)
    a, r = env(t, 1.4, d, 1.0, 1.2)
    put(im, "Check before you forward.", font("reg", 54), H / 2 + 110, TEAL, a, r)
    a, _ = env(t, 2.2, d, 1.0, 1.2)
    put(im, "An automated check. It can be wrong. Not medical, legal or financial advice.", font("reg", 26), H - 120, (150, 160, 165), a)
    put(im, "Narration: Kokoro text-to-speech. App footage partly sped up. Evidence: Wikipedia and news publishers.", font("reg", 22), H - 80, (105, 115, 120), a)
    return im


DOUBT = [
    Scene("It arrives on your phone. Forwarded many times.", s_forward),
    Scene("It sounds true. And everyone is sharing it.", s_sharing),
    Scene("But who actually checked?", s_question),
]
CLARITY = [
    Scene("Introducing Claim Checker.", s_title, min_dur=4.6),
    Scene(
        "Paste any forwarded message, or upload a screenshot.",
        ui=("home", "typed", "typed", 2.6), caption="Paste a message, or upload a screenshot.",
    ),
    Scene(
        "It splits the message into separate claims, and searches Wikipedia and trusted news.",
        ui=("clicked", "results", None, 0.0), caption="Claims split, and sources searched. (Sped up)", max_speedup=1000,
    ),
    Scene("Every claim gets a verdict. Supported, refuted, or unverified.", s_verdicts, min_dur=5.0),
    Scene(
        "Here, it catches a hoax, with the evidence linked, so you can read it for yourself.",
        ui=("results", "hoax", "hoax", 2.4), caption="A hoax, refuted, with the evidence linked.",
    ),
    Scene(
        "Open the evidence, and you see every source it found, with the publisher and the date.",
        ui=("hoax", "evidence", "evidence", 2.6), caption="Every source, with its publisher and date.",
    ),
    Scene(
        "Under the hood, a DeBERTa model scores how strongly each source supports or contradicts the claim. "
        "You can even see where its attention goes.",
        ui=("evidence", "model", "model", 3.0), caption="Under the hood: probabilities and attention.",
    ),
    Scene(
        "And when it can't find confident evidence, it says so, instead of guessing.",
        ui=("model", "unsure", "unsure", 2.4), caption="No confident evidence? It says so.",
    ),
    Scene(
        "Then share the result, as a card your whole group can read.",
        ui=("unsure", "card", "card", 2.8), caption="A card to send back to the group.",
    ),
    Scene(
        "Here's how it works. The message is read, split into claims by a language model, "
        "searched across Wikipedia and the news, and judged by a separate classifier.",
        s_pipeline, min_dur=9.0,
    ),
    Scene(
        "The same steps are explained inside the app, so anyone can see why a verdict was given.",
        ui=("how_start", "how", "how", 2.2), caption="The method, explained in the app.",
    ),
    Scene(
        "Got a screenshot instead? Upload it. The app reads the text with OCR, then checks it the same way.",
        ui=("shot_start", "shot_text", "shot_text", 2.6), caption="Screenshots work too. (Sped up)", max_speedup=1000,
    ),
    Scene("Phone numbers and emails are removed before anything reaches the AI.", s_privacy, min_dur=6.0),
    Scene(
        "On twelve claims it was never tuned on, it got ten right. And every mistake is written down.",
        s_score, min_dur=7.5,
    ),
    Scene("Every claim, checked. Every source, linked.", s_pillars, min_dur=6.8),
    Scene("Claim Checker. Check before you forward.", s_end, min_dur=8.0),
]
NARRATOR, PACE = "am_fenrir", 0.95  # the voice and pace used for the Helper in Law film


def voice(kokoro_dir: Path, scenes, work: Path):
    from kokoro_onnx import Kokoro

    k = Kokoro(str(kokoro_dir / "kokoro-v1.0.onnx"), str(kokoro_dir / "voices-v1.0.bin"))
    clips = []
    for i, sc in enumerate(scenes):
        pcm, sr = k.create(sc.vo, voice=NARRATOR, speed=PACE, lang="en-us")
        pcm = np.asarray(pcm, dtype=np.float64)
        with wave.open(str(work / f"vo_{i:02d}.wav"), "wb") as wf:  # kept for listening back
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes((np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes())
        clips.append(np.interp(np.arange(int(len(pcm) * SR / sr)) / SR, np.arange(len(pcm)) / sr, pcm))
    return clips


def durations(scenes, lens_, marks):
    """The problem takes exactly TURN seconds, so that picture and music turn together."""
    n = len(DOUBT)
    if sum(lens_[:n]) + 0.45 * n > TURN:
        raise SystemExit(f"opening narration is {sum(lens_[:n]):.1f}s: too long for the {TURN}s turn")
    raw = [max(sc.min_dur, x + 1.0) for sc, x in zip(DOUBT, lens_[:n], strict=True)]
    durs = [d * TURN / sum(raw) for d in raw]
    for sc, x in zip(CLARITY, lens_[n:], strict=True):
        d = max(sc.min_dur, x + 1.2)
        if sc.ui:
            d = max(d, (marks[sc.ui[1]]["t"] - marks[sc.ui[0]]["t"]) / sc.max_speedup)
        durs.append(d)
    return durs


def render(scenes, durs, marks, rec: Path, silent: Path) -> None:
    enc = encoder(silent)
    for sc, dur in zip(scenes, durs, strict=True):
        n = int(round(dur * FPS))
        print(f"{dur:5.1f}s {sc.vo[:60]}", flush=True)
        if sc.ui:
            a, b, focus, lead = sc.ui
            t0, t1 = marks[a]["t"], marks[b]["t"]
            box = marks[focus]["box"] if focus else None
            for k, (fr, src_t) in enumerate(screen_frames(rec / "screen.webm", t0, t1, n)):
                zk = ease((src_t - (t1 - lead)) / 1.1)
                gk = ease((src_t - (t1 - lead) - 1.0) / 0.6)
                enc.stdin.write(framed(fr, k / FPS, dur, sc.caption, box, zk, gk).tobytes())
        else:
            for k in range(n):
                enc.stdin.write(sc.render(k / FPS, dur).tobytes())
    enc.stdin.close()
    enc.wait()


def stills(scenes, durs, marks, rec: Path, work: Path) -> None:
    for i, (sc, dur) in enumerate(zip(scenes, durs, strict=True)):
        if sc.ui:
            a, b, focus, lead = sc.ui
            t0, t1 = marks[a]["t"], marks[b]["t"]
            box = marks[focus]["box"] if focus else None
            n = int(round(dur * FPS))
            fr, src_t, k = None, 0.0, 0
            for k, (fr, src_t) in enumerate(screen_frames(rec / "screen.webm", t0, t1, n)):
                pass  # the last frame shows the zoom at its fullest
            im = framed(fr, k / FPS, dur, sc.caption, box, ease((src_t - (t1 - lead)) / 1.1), ease((src_t - (t1 - lead) - 1.0) / 0.6))
        else:
            im = sc.render(dur * 0.6, dur)
        im.resize((960, 540)).save(work / f"still_{i:02d}.png")


def main(rec: Path, model: Path, track: Path, out: Path, work: Path, only_stills: bool) -> None:
    marks = json.loads((rec / "marks.json").read_text())
    work.mkdir(parents=True, exist_ok=True)
    scenes = DOUBT + CLARITY
    clips = voice(model, scenes, work)
    durs = durations(scenes, [len(c) / SR for c in clips], marks)
    total = sum(durs)
    print(f"total {total:.1f}s", flush=True)
    if only_stills:
        stills(scenes, durs, marks, rec, work)
        return
    silent = work / "film.silent.mp4"
    render(scenes, durs, marks, rec, silent)

    starts = [sum(durs[:i]) for i in range(len(durs))]
    music = music_bed(track, total, TURN)
    vo = np.zeros(len(music))
    for st, clip in zip(starts, clips, strict=True):
        i = int((st + 0.45) * SR)
        j = min(len(vo), i + len(clip))
        vo[i:j] += clip[: j - i]
    speech = np.abs(vo) > 0.01
    music *= 0.30 * np.sqrt(np.mean(vo[speech] ** 2)) / np.sqrt(np.mean(music**2))
    duck = 1 - 0.45 * np.clip(smooth(speech.astype(float), int(0.3 * SR)) * 3, 0, 1)
    wav = work / "film.wav"
    write_wav(wav, finish(vo[:, None] + music * duck[:, None]))
    out.parent.mkdir(parents=True, exist_ok=True)
    mux(silent, wav, out)
    print(f"wrote {out} ({total:.1f}s; turn at {TURN}s)")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out_path = Path(args[3])
    main(Path(args[0]), Path(args[1]), Path(args[2]), out_path,
         Path(args[4]) if len(args) > 4 else out_path.parent / "film_work", "--stills" in sys.argv)
