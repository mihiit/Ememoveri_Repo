# Model Configuration for E5-P (Multi-Model Evaluation)

This is the single source of truth for which LLMs back the E5 coding-agent
evaluation. Update this file first when the model set changes, then keep
`README.md` and `E5_INTEGRATION_GUIDE.md` pointing at it rather than
duplicating the table.

## Why this file exists

The blueprint's Non-Negotiable Rule #9 says: *"No single-model/single-
percentage conclusion — use paired comparisons, uncertainty and multiple
models where feasible."* A vague model list ("Llama", "Qwen", "Groq") is not
reproducible — "Groq" is an inference provider, not a model, and "Llama" /
"Qwen" span many versions with materially different capabilities. Every
model used in a real E5 run must be pinned here by exact provider API ID.

## Selected models (free tier, no credit card required)

All three run through [Groq](https://console.groq.com) — a free,
no-credit-card developer tier with generous per-organization rate limits.
No local GPU or HuggingFace download is required for these.

| Role | Model | Groq API ID | Family | Context | License |
|---|---|---|---|---|---|
| Model family 1 (primary) | Llama 3.3 70B Versatile | `llama-3.3-70b-versatile` | Meta / Llama | 131K tokens | Llama 3.3 Community License |
| Model family 2 (cross-family) | Qwen3 32B | `qwen/qwen3-32b` | Alibaba / Qwen | — (see caveat below) | Apache 2.0 |
| Model family 3 (smaller/faster, optional) | GPT-OSS-20B | `openai/gpt-oss-20b` | OpenAI open-weight | — | Apache 2.0 |

This deliberately replaces an earlier plan to include a small/weak model
(GPT-2, then a 1.5B Qwen-Coder variant) as a "weak baseline." All three
models above are current, genuinely capable, free, and cross-family — there
is no need to caveat any of them as deliberately weak, which makes for a
cleaner E5-P section than a small-model-vs-strong-model framing would.

## Reproducibility caveats (put these in the paper's Methods / Threats sections)

- **Qwen3 32B and any Llama 4-family model are marked "Preview" on Groq
  as of this writing.** Preview models can change or be silently
  deprecated by the provider. Record the exact date every E5 run was
  executed, and re-verify the model card / API ID immediately before a
  large-scale run. This maps directly onto the blueprint's own
  "Infrastructure" threat-to-validity category.
- Pin decoding parameters (temperature, top_p, max_tokens) identically
  across all three models and all four memory conditions per task — the
  blueprint requires that only the memory mechanism differ between
  conditions, and that requirement extends across models too (fix
  everything else, vary the model, run each model through the full
  four-condition ladder).
- Groq's free tier is rate-limited per organization, not per model — if you
  run all three models' full task sets back-to-back from one account,
  budget wall-clock time accordingly (see Groq's dashboard for current
  request-per-minute limits, which change independently of this document).
- Because these are hosted, closed-infrastructure endpoints (not weights
  you control), exact output can drift if the provider updates the
  underlying weights without changing the API ID. Log the response's
  provider-reported model version/build identifier (if Groq exposes one in
  the API response) alongside every task result for full reproducibility.

## How this maps onto the four-way baseline ladder

Every model above must be run through **all four** memory conditions in
`ememoveri/baselines.py` (`NO_MEMORY` / `PLAIN_RAG` / `NAIVE_MEMORY` /
`EMEMOVERI`) on the **same task set**, so that the resulting comparison
table (blueprint's Table 8, "per-model results") shows, for each model
independently:

```
model            | condition      | correct_rate | contamination_rate | coverage | ...
llama-3.3-70b... | NO_MEMORY      | ...
llama-3.3-70b... | PLAIN_RAG      | ...
llama-3.3-70b... | NAIVE_MEMORY   | ...
llama-3.3-70b... | EMEMOVERI      | ...
qwen/qwen3-32b   | NO_MEMORY      | ...
...
```

Report **both** absolute per-model performance and the paired
EmemoVeri-minus-baseline delta per model separately — do not collapse
across models into one headline number (blueprint's Non-Negotiable Rule #9
again).

## Where to plug this in

See `docs/E5_INTEGRATION_GUIDE.md` §4 ("Agent harness wiring") for where the
model call happens in the pipeline, and §5 for the exact `MemoryStore`
configuration per condition. The model itself is orthogonal to that wiring
— swap the API ID/provider call, keep everything else fixed.
