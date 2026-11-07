"""Offline self-check of the logic most likely to fail silently. Run: python test_checker.py"""
import io

from PIL import Image

import checker


def ev(entailment, contradiction, relevance=0.8, sentence="A plain sentence.", title="Page", kind="wikipedia"):
    return {"entailment": entailment, "contradiction": contradiction, "relevance": relevance, "sentence": sentence, "title": title, "kind": kind}


def verdict(*evidence):
    return checker.decide(list(evidence))[0]


assert checker.decide([]) == ("unverified", 0.0, None)
assert verdict(ev(0.5, 0.2)) == "unverified"
assert verdict(ev(0.95, 0.01)) == "supported"
assert verdict(ev(0.01, 0.95)) == "refuted"
# A closer entailing sentence beats a confident contradiction from a tangent, and the reverse.
assert verdict(ev(0.98, 0.0, 0.86), ev(0.0, 0.99, 0.77)) == "supported"
assert verdict(ev(0.9, 0.0, 0.62), ev(0.0, 0.99, 0.85)) == "refuted"
# Contradiction from a barely related sentence is ignored.
assert verdict(ev(0.0, 1.0, 0.41)) == "unverified"
# Reported beliefs count against the claim, whether the cue is in the sentence or the page title.
assert verdict(ev(0.92, 0.0, sentence="The myth states that humans use 10%.")) == "refuted"
assert verdict(ev(0.99, 0.0, title="Ten-percent-of-the-brain myth")) == "refuted"
# The deciding sentence is returned so the UI can cite it.
closer = ev(0.9, 0.0, 0.9)
assert checker.decide([ev(0.9, 0.0, 0.5), closer])[2] is closer

# A publisher's name must not count as a myth marker, unlike a Wikipedia page title.
assert verdict(ev(0.95, 0.0, sentence="Taj Mahal is in Agra.", title="FactCheck Myths", kind="news")) == "supported"
assert verdict(ev(0.95, 0.0, sentence="Viral claim that X is false.", title="Reuters", kind="news")) == "refuted"

FEED = b"""<rss><channel>
<item><title>No, 5G did not spread the virus - Reuters</title><link>https://news.google.com/rss/articles/abc?x=1</link>
  <pubDate>Fri, 04 Sep 2020 07:00:00 GMT</pubDate><source url="https://www.reuters.com">Reuters</source></item>
<item><title>5G spreads the virus, say experts - Random Blog</title><link>https://news.google.com/rss/articles/def</link>
  <source url="https://notreuters.com">Random Blog</source></item>
<item><title>Did 5G spread the virus at all? - BBC</title><link>https://news.google.com/rss/articles/ghi</link>
  <source url="https://www.bbc.com">BBC</source></item>
<item><title>A bad link for this headline - BBC</title><link>https://evil.example/x)[y</link>
  <source url="https://www.bbc.com">BBC</source></item>
<item><title>No date on this headline here - BBC</title><link>https://news.google.com/rss/articles/jkl</link>
  <source url="https://www.bbc.com">BBC</source></item>
</channel></rss>"""
items = checker.parse_news_feed(FEED)
assert [i["sentence"] for i in items] == ["No, 5G did not spread the virus", "No date on this headline here"], items
assert items[0]["label"] == "Reuters, 04 Sep 2020" and items[1]["label"] == "BBC, date unknown"
assert not checker.is_trusted("https://bbc.com.evil.example") and checker.is_trusted("https://factcheck.afp.com")


def fails(_):
    raise checker.requests.RequestException("down")


# One source failing must not stop the check, but all of them failing must be reported.
real_sources = dict(checker.SOURCES)
checker.SOURCES.update(Wikipedia=fails, News=lambda _: [])
assert checker.retrieve("a claim") == ([], ["Wikipedia"])
checker.SOURCES.update(News=fails)
try:
    checker.retrieve("a claim")
except checker.requests.RequestException:
    pass
else:
    raise AssertionError("expected an error when every source fails")
checker.SOURCES.update(real_sources)

# Replies from reasoning models may wrap the JSON in thinking or prose.
assert checker.parse_json_reply('{"claims": ["a"]}') == {"claims": ["a"]}
assert checker.parse_json_reply('<think>{"x": 1}</think>Here it is: {"claims": []} done.') == {"claims": []}
try:
    checker.parse_json_reply("no json here")
except ValueError:
    pass
else:
    raise AssertionError("expected ValueError for a reply without JSON")

redacted = checker.redact("Call +91 98765 43210 or mail a.b@example.com by 1948")
assert "98765" not in redacted and "example.com" not in redacted and "1948" in redacted

assert checker.search_query("The Moon is made of cheese") == "Moon OR made OR cheese"

old_key = checker.os.environ.pop("GROQ_API_KEY", None)
assert checker.llm_config()["provider"] == "ollama"
checker.os.environ["GROQ_API_KEY"] = "test"
assert checker.llm_config()["provider"] == "groq"
checker.os.environ.pop("GROQ_API_KEY")
if old_key:
    checker.os.environ["GROQ_API_KEY"] = old_key

result = {"claim": "A claim " * 60, "verdict": "refuted", "confidence": 0.9, "evidence": [{"label": "Wikipedia: Page"}]}
full = Image.open(io.BytesIO(checker.make_card([result] * 12)))
short = Image.open(io.BytesIO(checker.make_card([result])))
assert full.width == short.width == 1080 and short.height < full.height <= 1350

print("all checks passed")
