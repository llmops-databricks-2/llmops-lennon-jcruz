# Sanitized failure example — `grounded_evidence` on High Risk Proxy PD

**Eval run:** `1725406606074df89e0fab486990577c` (job `mra-agent-eval`, after judge-model fix)  
**Trace id (workspace):** `trace:/llmops_dev.lennon_jcruz.llmops/3d83be9fd6a46011358ecbf9dc237de3`  
**Case:** `high_risk_unfair_features` / model `High_Risk_Proxy_PD`  
**Agent decision:** `deny` (matches expected band)  
**`grounded_evidence`:** `no` (mean contribution to 0.0 across the run)

Paths, emails, and raw UUIDs below are redacted. Chunk ids keep the real corpus naming pattern so the failure is reproducible.

---

## Trace tree (sanitized)

```
AGENT assess
├── RETRIEVER memory_load
├── CHAIN chat
│   ├── LLM gateway_chat
│   ├── TOOL execute_tool  (search_regulatory)
│   │   └── EMBEDDING …
│   ├── LLM gateway_chat
│   ├── TOOL execute_tool  (search_regulatory / follow-up)
│   │   └── EMBEDDING …
│   ├── LLM gateway_chat
│   ├── TOOL execute_tool
│   │   └── EMBEDDING …
│   └── LLM gateway_chat  → final report
└── CHAIN memory_save
```

Tags observed: `run_id`, `model_name=High_Risk_Proxy_PD`, `decision=deny`, `git_sha=local`, `mlflow.eval.requestId=<redacted>`.

---

## Tool evidence that *was* retrieved (excerpt)

`execute_tool` returned real Lakebase hits, including:

```json
[
  {
    "id": "frb-sr-11-7-attachment_53",
    "title": "Supervisory Guidance on Model Risk Management (SR 11-7 attachment)",
    "corpus": "regulatory",
    "score": 0.6812,
    "excerpt": "Model validation is the set of processes and activities intended to verify…"
  },
  {
    "id": "occ-bulletin-2011-12_67",
    "title": "OCC Bulletin 2011-12 — Sound Practices for Model Risk Management",
    "corpus": "regulatory",
    "score": 0.6709,
    "excerpt": "Model validation is the set of processes and activities…"
  }
]
```

So the agent did call tools and received titled regulatory chunks with ids.

---

## Agent Evidence section (excerpt)

```markdown
## Evidence
- **Supervisory Guidance on Model Risk Management (SR 11-7 attachment)** — …
  【frb-sr-11-7-attachment_53】.
- **OCC Bulletin 2011-12 – Sound Practices for Model Risk Management** — …
  【occ-bulletin-2011-12_112】.
```

Findings cite those ids, but bullets do **not** use the prompt’s preferred shape  
`[corpus] title (id=…)` — corpus is missing as an explicit field in the Evidence list.

---

## Judge rationale (condensed)

Guidelines `grounded_evidence` scored `no` because:

1. It only sees eval **inputs** (deck + question) and **outputs** (report text) — not TOOL spans.
2. Without tool payloads in the judge context, retrieval ids look “invented” / unverifiable.
3. Guideline 3 asks for `corpus/title/id`; the report had title + id but not an explicit corpus token.

This is a **scorer observability / Evidence-format** failure, not “agent skipped retrieval.”
