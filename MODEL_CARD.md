# Model Card: Claim Checker for Viral Messages

This app combines existing models and does not train a new one. No fine-tuning was done.

## Models

| Role | Model | Type | Licence |
| --- | --- | --- | --- |
| Claim extraction (hosted) | `openai/gpt-oss-120b` on Groq | Generative, GPT-style transformer | Apache 2.0 |
| Claim extraction (local) | `qwen3:1.7b` on Ollama | Generative, GPT-style transformer | Apache 2.0 |
| Verification | `cross-encoder/nli-deberta-v3-small` | Discriminative, BERT family | Apache 2.0 |
| Evidence ranking and grounding | `sentence-transformers/all-MiniLM-L6-v2` | BERT-family sentence embeddings | Apache 2.0 |
| Screenshot OCR | EasyOCR (CRAFT detector, CRNN recogniser) | Vision, PyTorch | Apache 2.0 |

Check the licence on each model's own page before reusing this work outside coursework.

## Intended use

Giving a first opinion on English forwarded messages about stable, well-documented facts, with the evidence
shown so a person can judge it.

## Out of scope

- Medical, legal or financial decisions.
- Breaking news that fact-checkers haven't covered yet, local rumours, and claims about private individuals.
- Languages other than English. The NLI model, the OCR setting and the card font are all English or Latin only.

## How a verdict is made

The LLM only splits the message into claims and never decides a verdict. For each claim, the app retrieves
Wikipedia sentences and news headlines, and DeBERTa scores each (evidence, claim) pair. A verdict needs a
probability of at least 0.75. Supported or refuted comes from the item with the best probability × relevance.
Anything else is unverified. When confident items point both ways, the app warns that sources disagree, but the
verdict itself doesn't change.

## Evidence sources

- **Wikipedia**, through its search API, up to three pages per claim.
- **News**, through Google News RSS, searched twice per claim: once plainly and once for fact-checks. Only
  publishers on the `TRUSTED_NEWS` allowlist in `checker.py` are used. Headlines that only ask a question are ignored.

The allowlist is the author's own judgement, not an independent rating. Choosing it is itself an editorial decision,
and being listed doesn't make a publisher right.

## Evaluation

Measured numbers are in [results/evaluation.md](results/evaluation.md), and they include:

- accuracy on a tuning set and on a separate held-out set
- a comparison of zero-shot and few-shot prompts
- a hallucination test of the LLM answering with no evidence

Thresholds were tuned by inspecting the tuning set, so the held-out number is the more honest one. Both sets
are small and hand-labelled by the author, so the numbers are indicative only.

## Known limitations

- DeBERTa over-predicts contradiction on text that is on-topic but says something different. The relevance
  thresholds reduce this and don't remove it.
- The reported-belief rule is a keyword list. A sentence that reports a false belief without a word like "myth"
  or "claim" can count as support, and an accurate sentence containing such a word can count against a claim.
- A verdict can be right for the wrong reason, because search sometimes returns pages that only share words with
  the claim. Always read the linked evidence.
- Wikipedia is an editable secondary source and can itself be wrong or out of date.
- News evidence is the headline only, not the article. Headlines simplify, and they can be years old, so a true
  statement from 2019 can be out of date now. Dates are shown, but the model doesn't take them into account.
- The same story republished by related outlets counts as several evidence items, which can overstate agreement.
- Google News RSS is not a documented, supported API, so it could change or start limiting requests without notice.
  If it does, the app carries on with Wikipedia and says that news wasn't checked.
- Unverified means no confident evidence was found. It doesn't mean the claim is false.

## Safety and privacy

- Emails and phone numbers are redacted before message text reaches the LLM. This is regex-based, so names and
  other personal details are not removed.
- With Groq, the redacted message text is sent to a third party. With Ollama, it stays on the machine.
- Message text is passed as data inside tags, and the prompt tells the LLM to ignore instructions in it. The LLM
  has no say in the verdict, so an injected "mark everything true" can't change one directly.
- Extracted claims that don't resemble the message are dropped, which guards against invented claims.

## Bias and fairness

Results reflect the coverage of Wikipedia and the allowlisted outlets, which are mostly English-language and either
Indian or Western. Claims about regional events, or from other parts of the world, are more likely to come back
unverified, even when they are true or false in a way that is easy to establish locally. This was not measured.

## IP and licensing

Wikipedia text is CC BY-SA, so evidence is shown as short quotes with links to the source page. News headlines
belong to their publishers and are shown with attribution and a link for fact-checking purposes. Check Google's
terms before any commercial use of its RSS feeds. Users are responsible for rights in screenshots they upload. The app doesn't store messages or uploads.
