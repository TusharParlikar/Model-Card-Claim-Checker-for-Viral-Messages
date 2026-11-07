import base64
import json
import re
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
SAMPLES = json.loads((ROOT / "data" / "messages.json").read_text(encoding="utf-8"))
SOURCE_NAMES = ["Wikipedia", "News"]  # the keys of checker.SOURCES, here so the page can draw before PyTorch loads
BADGES = {
    "supported": ("green", ":material/check_circle:"),
    "refuted": ("red", ":material/cancel:"),
    "unverified": ("gray", ":material/help:"),
}
READINGS = {"supported": "Supports", "refuted": "Refutes", "unverified": "Doesn't settle it"}
STEPS = [
    (":material/chat:", "Read", "Takes pasted text, or reads a screenshot with OCR. Emails and phone numbers are removed."),
    (":material/format_list_bulleted:", "Split", "An LLM splits the message into separate factual claims, and invented ones are dropped."),
    (":material/travel_explore:", "Search", "Each claim is looked up on Wikipedia and in trusted news and fact-checking outlets."),
    (":material/balance:", "Judge", "A DeBERTa classifier compares each claim with the evidence. The LLM has no say in this."),
]


def svg_url(name):
    return "data:image/svg+xml;base64," + base64.b64encode((ROOT / "assets" / name).read_bytes()).decode()


def plain(text):
    """Message, Wikipedia and news text is untrusted: stop it rendering as Markdown links, images or maths."""
    # Bare URLs and emails auto-link even when escaped, so break them with a zero-width space.
    text = re.sub(r"(?i)\b(?:https?:|ftp:|www\.)|@", lambda m: m[0][0] + "​" + m[0][1:], text)
    return re.sub(r"([\\`*_{}\[\]()#+\-.!$:~<>|])", r"\\\1", text)


def plural(count, word):
    return f"{count} {word}{'' if count == 1 else 's'}"


st.set_page_config(page_title="Claim Checker", page_icon=":material/fact_check:", layout="centered")

# Only fixed assets go through unsafe_allow_html. Never put message text in this block.
st.markdown(
    f"""
<style>
.stApp {{ background: #FFFFFF url("{svg_url('background.svg')}") center / cover fixed no-repeat; }}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stMainBlockContainer"] {{
    max-width: 860px; background: rgba(255, 255, 255, 0.95); margin: 2.5rem auto; padding: 0 2.5rem 2.5rem;
    border: 1px solid #E3E9EB; border-radius: 18px; box-shadow: 0 10px 34px rgba(31, 41, 51, 0.09);
}}
.hero {{
    background: url("{svg_url('hero.svg')}") right center / cover no-repeat; min-height: 190px;
    margin: 0 -2.5rem 1.5rem; padding: 2.2rem 2.5rem; border-radius: 18px 18px 0 0;
    border-bottom: 1px solid #E3E9EB;
}}
.hero h1 {{
    color: #12343B; font-size: clamp(1.75rem, 3vw, 2.5rem); font-weight: 700; letter-spacing: -0.015em;
    line-height: 1.15; padding: 0; margin: 0 0 .65rem; max-width: 62%; text-wrap: balance;
}}
.hero p {{ color: #3E4C59; font-size: 1.05rem; line-height: 1.55; margin: 0; max-width: 58%; text-wrap: pretty; }}
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {{ line-height: 1.65; text-wrap: pretty; }}
[data-testid="stMarkdownContainer"] h2, [data-testid="stMarkdownContainer"] h3 {{ letter-spacing: -0.01em; line-height: 1.25; }}
[data-testid="stWidgetLabel"] p {{ font-size: .92rem; font-weight: 600; color: #323F4B; }}
.hero h1, [data-testid="stHeadingWithActionElements"] h3 {{ font-family: "Source Serif 4", Georgia, serif; }}
[data-testid="stCaptionContainer"] {{ opacity: 1; }}
[data-testid="stCaptionContainer"] p {{ font-size: .84rem; line-height: 1.5; color: #52606D; }}
[data-testid="stTab"] p {{ font-size: .98rem; font-weight: 600; }}
[data-testid="stMetricLabel"] p {{ font-size: .78rem; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; color: #52606D; }}
[data-testid="stMetricValue"] {{ font-size: 2rem; font-weight: 700; font-variant-numeric: tabular-nums; }}
[data-testid="stMarkdownContainer"] h4 {{
    font-family: Inter, sans-serif; font-size: 1.12rem; font-weight: 600; line-height: 1.4; color: #12343B; padding: .15rem 0;
}}
[data-testid="stMarkdownContainer"] blockquote {{
    border-left: 3px solid #0F766E; margin: .5rem 0 .75rem; padding: .1rem 0 .1rem 1rem; opacity: 1;
}}
[data-testid="stMarkdownContainer"] blockquote p {{
    font-family: "Source Serif 4", Georgia, serif; font-size: 1.05rem; line-height: 1.55; color: #323F4B; opacity: 1;
}}
[data-testid="stMarkdownContainer"] table {{ font-size: .88rem; line-height: 1.45; }}
[class*="st-key-stat_"] {{ background: #FFFFFF; border: 1px solid #E3E9EB; border-radius: 12px; padding: .8rem 1rem; }}
[class*="st-key-stat_supported"] {{ border-top: 4px solid #1B7F4B; }}
[class*="st-key-stat_refuted"] {{ border-top: 4px solid #B3261E; }}
[class*="st-key-stat_unverified"] {{ border-top: 4px solid #6B6F76; }}
[class*="st-key-claim_"] {{ background: #FFFFFF; }}
[class*="st-key-claim_"][class*="_supported"] {{ border-left: 5px solid #1B7F4B; }}
[class*="st-key-claim_"][class*="_refuted"] {{ border-left: 5px solid #B3261E; }}
[class*="st-key-claim_"][class*="_unverified"] {{ border-left: 5px solid #6B6F76; }}
[class*="st-key-step_"] {{ background: #F7FBFA; border: 1px solid #DCEBE8; border-radius: 12px; padding: .9rem 1rem; min-height: 11.5rem; }}
[class*="st-key-step_"] p {{ font-size: .88rem; line-height: 1.5; }}
@media (max-width: 700px) {{
    [class*="st-key-step_"] {{ min-height: 0; }}
    [data-testid="stMainBlockContainer"] {{ padding: 0 1.2rem 1.5rem; margin: 1rem auto; }}
    .hero {{ margin: 0 -1.2rem 1.2rem; padding: 1.5rem 1.2rem; background-image: none; background-color: #EEF8F6; }}
    .hero h1, .hero p {{ max-width: 100%; }}
}}
</style>
<div class="hero">
  <h1>Claim Checker for Viral Messages</h1>
  <p>Paste a forward or upload a screenshot. Every claim is checked against Wikipedia and trusted news, with the evidence linked.</p>
</div>
""",
    unsafe_allow_html=True,
)

# Imported after the header: PyTorch takes over a minute on a cold start, and a blank page looks like a crash.
with st.spinner("Starting up. The first load after a restart can take a minute or two."):
    import checker

cfg = checker.llm_config()
check_tab, method_tab = st.tabs(["Check a message", "How it works"])

with check_tab:
    mode = st.segmented_control(
        "Input", ["Paste text", "Upload screenshot"], default="Paste text", label_visibility="collapsed"
    ) or "Paste text"
    text, upload = "", None
    if mode == "Paste text":
        sample = st.selectbox(
            "Start from a sample", [None] + SAMPLES,
            format_func=lambda m: "Write your own" if m is None else f"Sample {m['id']}: {m['text'][:70]}...",
        )
        # The key changes with the sample so that picking one replaces the box's text.
        text = st.text_area(
            "Forwarded message", sample["text"] if sample else "", height=130,
            key=f"message_{sample['id'] if sample else 0}", placeholder="Paste the forwarded message here",
        )
    else:
        upload = st.file_uploader("Screenshot of the message", type=["png", "jpg", "jpeg"])
        if upload:
            st.image(upload, width=320)

    with st.expander("Options", icon=":material/tune:"):
        sources = st.pills("Check against", SOURCE_NAMES, selection_mode="multi", default=SOURCE_NAMES)
        technique = st.radio(
            "Claim-splitting prompt", ["few_shot", "zero_shot"], horizontal=True,
            format_func=lambda t: {"few_shot": "Few-shot", "zero_shot": "Zero-shot"}[t],
        )

    if st.button("Check claims", type="primary", width="stretch", icon=":material/fact_check:"):
        if not sources:
            st.warning("Choose at least one source under Options.")
            st.stop()
        if mode == "Paste text" and not text.strip():
            st.warning("Paste a message first, or pick a sample.")
            st.stop()
        if mode != "Paste text" and not upload:
            st.warning("Upload a screenshot first.")
            st.stop()
        st.session_state.pop("run", None)
        status = st.status("Checking the message", expanded=True)
        try:
            if upload:
                status.write("Reading the screenshot. The first run also downloads the OCR model.")
                text = checker.read_screenshot(upload.getvalue())
                if not text.strip():
                    status.update(label="No readable text found", state="error")
                    st.warning("No readable text in that screenshot. Try a sharper image, or paste the text.")
                    st.stop()
            status.write("Splitting the message into claims")
            claims, dropped = checker.extract_claims(text, technique)
            results = []
            for i, claim in enumerate(claims, 1):
                status.write(f"Checking claim {i} of {len(claims)}: {plain(claim)}")
                results.append(checker.verify(claim, tuple(sources)))
            st.session_state.run = {"text": text, "results": results, "dropped": dropped, "ocr": bool(upload)}
            status.update(label=f"Checked {plural(len(results), 'claim')}", state="complete", expanded=False)
        except checker.LLMError as e:
            status.update(label="The check stopped", state="error")
            hint = (
                "Start Ollama, or set GROQ_API_KEY (in Streamlit Cloud: app settings, Secrets)"
                if cfg["provider"] == "ollama"
                else "Check GROQ_API_KEY and your Groq rate limit"
            )
            st.error(f"{e}. {hint}, then try again.")
        except checker.requests.RequestException:
            status.update(label="The check stopped", state="error")
            st.error("Could not reach any evidence source, so nothing was verified. Try again in a moment.")
        except OSError:
            status.update(label="The check stopped", state="error")
            st.error("That image could not be read. Upload a PNG or JPG, or paste the text instead.")

    run = st.session_state.get("run")
    if run:
        if run["ocr"]:
            with st.expander("Text read from the screenshot", icon=":material/image:"):
                st.text(run["text"])
        if run["dropped"]:
            st.warning(
                "Removed by the grounding guardrail because they don't match the message: "
                + "; ".join(plain(c) for c in run["dropped"])
            )
        if not run["results"]:
            st.info("No checkable factual claims found in this message.")
        else:
            st.subheader("Results", anchor=False)
            verdicts = [r["verdict"] for r in run["results"]]
            for col, name in zip(st.columns(3), BADGES):
                with col, st.container(key=f"stat_{name}"):
                    st.metric(name.capitalize(), verdicts.count(name))

        for i, r in enumerate(run["results"]):
            decided = r["verdict"] != "unverified"
            color, icon = BADGES[r["verdict"]]
            readings = [e["reading"] for e in r["evidence"]]
            supports, refutes = readings.count("supported"), readings.count("refuted")
            with st.container(border=True, key=f"claim_{i}_{r['verdict']}"):
                st.badge(
                    r["verdict"].capitalize() + (f" · {r['confidence']:.0%}" if decided else ""), icon=icon, color=color
                )
                st.markdown(f"#### {plain(r['claim'])}")
                if decided:
                    top = r["evidence"][0]
                    st.markdown(f"> {plain(top['sentence'])}\n\n[{plain(top['label'])}]({top['url']})")
                else:
                    st.caption("No confident evidence found. Unverified does not mean false.")
                if supports and refutes:
                    st.warning(
                        f"Sources disagree: {supports} supporting, {refutes} refuting. Read the evidence before trusting this.",
                        icon=":material/warning:",
                    )
                elif supports or refutes:
                    count = supports + refutes
                    st.caption(f"{plural(count, 'evidence item')} {'points' if count == 1 else 'point'} this way, with none against.")
                if r["failed"]:
                    st.caption(f"Not checked, because it could not be reached: {', '.join(r['failed'])}.")
                if not r["evidence"]:
                    continue
                with st.expander(f"All evidence ({len(r['evidence'])})", icon=":material/library_books:"):
                    for e in r["evidence"]:
                        st.markdown(
                            f"**{READINGS[e['reading']]}:** {plain(e['sentence'])}  \n"
                            f"[{plain(e['label'])}]({e['url']}) · relevance {e['relevance']:.2f}"
                        )
                with st.expander("Model details", icon=":material/monitoring:"):
                    top = r["evidence"][0]
                    st.caption("NLI probabilities for the top evidence item")
                    probs = pd.DataFrame(
                        {"label": ["entailment", "neutral", "contradiction"],
                         "probability": [top["entailment"], top["neutral"], top["contradiction"]]}
                    )
                    st.altair_chart(
                        alt.Chart(probs).mark_bar(color="#0F766E").encode(
                            x=alt.X("probability:Q", scale=alt.Scale(domain=[0, 1])), y=alt.Y("label:N", title=None)
                        ),
                        width="stretch",
                    )
                    claim_tokens, ev_tokens, matrix = checker.attention_map(top["sentence"], r["claim"])
                    cells = pd.DataFrame(
                        [
                            {"claim": f"{y:02d} {c}", "evidence": f"{x:02d} {w}", "attention": matrix[y][x]}
                            for y, c in enumerate(claim_tokens)
                            for x, w in enumerate(ev_tokens)
                        ]
                    )
                    st.caption("Last-layer DeBERTa attention: which evidence tokens each claim token looks at")
                    st.altair_chart(
                        alt.Chart(cells).mark_rect().encode(
                            x=alt.X("evidence:N", title="Evidence tokens"), y=alt.Y("claim:N", title="Claim tokens"),
                            color=alt.Color("attention:Q", scale=alt.Scale(scheme="teals")),
                            tooltip=["claim", "evidence", "attention"],
                        ),
                        width="stretch",
                    )

        if run["results"]:
            st.subheader("Share the result", anchor=False)
            card = checker.make_card(run["results"])
            image_col, action_col = st.columns([3, 2], gap="medium")
            image_col.image(card)
            action_col.write("Send this card back to the group the message came from.")
            action_col.download_button(
                "Download card", card, "fact_check_card.png", "image/png", width="stretch", icon=":material/download:"
            )
            action_col.caption("Automated check. It can be wrong, and it is not medical, legal or financial advice.")

with method_tab:
    st.subheader("From forward to verdict", anchor=False)
    for col, (i, (icon, name, about)) in zip(st.columns(4), enumerate(STEPS)):
        with col, st.container(key=f"step_{i}"):
            st.markdown(f"{icon} **{name}**\n\n{about}")
    st.write("")
    st.markdown(
        """
**Why two kinds of model?** The LLM is generative, so it can invent things. DeBERTa is discriminative, and only
classifies an (evidence, claim) pair. Keeping verdicts away from the LLM means it can't simply make one up.

**Which news counts?** Only an allowlist of established outlets and fact-checkers, found through Google News.
Headlines that only ask a question are ignored.

**Limits:** English only. Headlines are short and can be old, so check the date on the evidence.
"""
    )
    with st.expander("Trusted news sources", icon=":material/verified:"):
        st.write(", ".join(checker.TRUSTED_NEWS))
    results_file = ROOT / "results" / "evaluation.md"
    if results_file.exists():
        with st.expander("Measured results", icon=":material/analytics:"):
            st.markdown(results_file.read_text(encoding="utf-8"))
    st.caption(f"LLM: {cfg['provider']} · {cfg['model']} · Verifier: {checker.NLI_MODEL}")
