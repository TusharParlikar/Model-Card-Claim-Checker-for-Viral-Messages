"""Record the real app in a headless browser, for the promo film.

python promo/record.py OUT_DIR SCREENSHOT.png [APP_URL]
Writes OUT_DIR/screen.webm and OUT_DIR/marks.json: for each moment, seconds from the start of the
recording and the on-screen box of the element that matters then (for zooms and highlights).
"""

import json
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

W, H = 1600, 1000
MESSAGE = (
    "Proud moment! UNESCO has declared the Indian national anthem the best in the world. "
    "UNESCO is headquartered in Paris. "
    "A local shop in Pune gives free phones to anyone who shares this post."
)


def main(out: Path, screenshot: Path, app: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    marks: dict[str, dict[str, object]] = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        ctx = browser.new_context(
            viewport={"width": W, "height": H},
            record_video_dir=str(out),
            record_video_size={"width": W, "height": H},
        )
        t0 = time.monotonic()
        page = ctx.new_page()

        def mark(name: str, selector: str | None = None) -> None:
            box = None
            if selector:
                b = page.locator(selector).filter(visible=True).first.bounding_box()
                if b:
                    box = [round(b["x"]), round(b["y"]), round(b["width"]), round(b["height"])]
            marks[name] = {"t": round(time.monotonic() - t0, 2), "box": box}

        def mark_union(name: str, selector: str) -> None:
            boxes = [b for b in (x.bounding_box() for x in page.locator(selector).all()) if b]
            x0, y0 = min(b["x"] for b in boxes), min(b["y"] for b in boxes)
            x1 = max(b["x"] + b["width"] for b in boxes)
            y1 = max(b["y"] + b["height"] for b in boxes)
            marks[name] = {"t": round(time.monotonic() - t0, 2), "box": [round(x0), round(y0), round(x1 - x0), round(y1 - y0)]}

        def scroll_to(selector: str, seconds: float = 2.5) -> None:
            """Glide until the element sits near the top of the view, as a person would scroll."""
            target = page.locator(selector).filter(visible=True).first
            page.mouse.move(W / 2, H / 2)
            box = target.bounding_box()
            assert box, selector
            distance = box["y"] - 90
            steps = int(seconds * 30)
            for i in range(steps):
                page.mouse.wheel(0, distance / steps)
                page.wait_for_timeout(1000 / 30)

        page.goto(app, wait_until="commit")
        page.get_by_role("button", name="Check claims").wait_for(timeout=300000)
        page.wait_for_timeout(2500)
        mark("home")
        page.wait_for_timeout(1500)

        box = page.get_by_label("Forwarded message")
        box.click()
        box.press_sequentially(MESSAGE, delay=24)
        box.press("Control+Enter")
        page.wait_for_timeout(1200)
        mark("typed", "textarea")
        page.get_by_role("button", name="Check claims").click()
        page.wait_for_timeout(600)
        mark("clicked")
        page.get_by_role("button", name="Download card").wait_for(timeout=480000)
        page.wait_for_timeout(1800)
        mark("results", '[class*="st-key-stat_refuted"]')

        # The film makes specific statements about these, so fail here if the run didn't show them.
        for verdict in ("refuted", "supported", "unverified"):
            found = page.locator(f'[class*="st-key-claim_"][class*="_{verdict}"]').count()
            if not found:
                raise SystemExit(f"no {verdict} claim in this run: re-record")
        hoax = page.locator('[class*="st-key-claim_"][class*="_refuted"]').first.inner_text()
        if "UNESCO" not in hoax:
            raise SystemExit(f"the refuted claim isn't the hoax: {hoax[:200]}")
        print("hoax card:", hoax[:200].replace("\n", " | "))

        refuted = '[class*="st-key-claim_"][class*="_refuted"]'
        scroll_to(refuted)
        page.wait_for_timeout(2500)
        mark("hoax", refuted)
        page.wait_for_timeout(1500)

        hoax_card = page.locator(refuted).first
        hoax_card.get_by_text("All evidence").click()
        page.wait_for_timeout(1500)
        evidence = f'{refuted} [data-testid="stExpander"] >> nth=0'
        scroll_to(evidence, 2.0)
        page.wait_for_timeout(3000)
        mark("evidence", evidence)
        page.wait_for_timeout(1000)

        hoax_card.get_by_text("Model details").click()
        page.wait_for_timeout(3500)
        details = f'{refuted} [data-testid="stExpander"] >> nth=1'
        scroll_to(details, 2.5)
        page.wait_for_timeout(3000)
        mark("model", details)
        page.wait_for_timeout(1000)

        unverified = '[class*="st-key-claim_"][class*="_unverified"]'
        scroll_to(unverified, 2.0)
        page.wait_for_timeout(2500)
        mark("unsure", unverified)
        page.wait_for_timeout(1500)

        card = 'img[src*="media"]'
        scroll_to(card, 2.0)
        page.wait_for_timeout(3000)
        mark("card", card)
        page.wait_for_timeout(1500)

        # How it works.
        page.mouse.move(W / 2, H / 2)
        page.mouse.wheel(0, -30000)
        page.wait_for_timeout(1200)
        page.get_by_role("tab", name="How it works").click()
        page.wait_for_timeout(800)
        mark("how_start")
        page.wait_for_timeout(2500)
        mark_union("how", '[class*="st-key-step_"]')
        page.wait_for_timeout(1500)

        # A screenshot instead of text, in a fresh session so the page starts clean on the first tab.
        page.goto(app, wait_until="commit")
        page.get_by_role("button", name="Check claims").wait_for(timeout=300000)
        page.wait_for_timeout(1500)
        page.get_by_text("Upload screenshot", exact=True).click()
        page.wait_for_timeout(1200)
        mark("shot_start")
        page.locator('input[type="file"]').set_input_files(screenshot)
        page.wait_for_timeout(2500)
        page.get_by_role("button", name="Check claims").click()
        page.get_by_text("Reading the screenshot").wait_for(timeout=60000)
        page.get_by_role("button", name="Download card").wait_for(timeout=480000)
        page.wait_for_timeout(1500)
        page.get_by_text("Text read from the screenshot").click()
        page.wait_for_timeout(1200)
        ocr = '[data-testid="stExpander"] >> nth=0'
        scroll_to(ocr, 2.0)
        page.wait_for_timeout(2500)
        mark("shot_text", ocr)
        page.wait_for_timeout(1500)
        mark("end")

        video = page.video
        ctx.close()
        browser.close()
        if video:
            Path(video.path()).replace(out / "screen.webm")
    (out / "marks.json").write_text(json.dumps(marks, indent=1))
    print(json.dumps(marks))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "http://127.0.0.1:8502")
