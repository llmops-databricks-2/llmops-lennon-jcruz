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

---

## Fix applied (follow-up)

Code changes after this failure write-up:

1. **Evidence format (prompts)** — `SYSTEM_PROMPT` / `REPORT_TEMPLATE` require every Evidence bullet as
   `[corpus] title (id=<chunk_id>)` and forbid inventing ids not returned by tools.
2. **Trace-aware retrieval** — search tools emit nested `RETRIEVER` spans with MLflow `Document` outputs (`tools.py`), and eval adds `RetrievalGroundedness` plus a softened `grounded_evidence` Guidelines rubric that accepts the `[corpus] title (id=…)` form without needing raw tool payloads in the request.

**Re-eval run:** `3e8be9a86adb43509ddf6ee01d9ec1d8` (6 cases, including regression)

| Scorer | Mean (after fix) | Notes |
|--------|------------------|--------|
| `decision_label_check` | 1.0 | unchanged |
| `report_structure_check` | 1.0 | unchanged |
| `decision_policy` | 1.0 | unchanged |
| `report_sections` | 1.0 | unchanged |
| `grounded_evidence` | **0.17** | was **0.0** |
| `retrieval_groundedness` | **0.05** | new; still weak |

Groundedness improved only slightly — format/judge wiring helped a little; claim-vs-chunk alignment remains the hard part.

---

## Finding — tool mix (regulatory vs academic)

Counted `execute_tool` → `tool_name` on the same two eval runs (excluding `hybrid_search_mra`, which was **never** called):

| Eval run | `search_regulatory` | `search_academic` | hybrid |
|----------|---------------------|-------------------|--------|
| Before fix `17254066…` (5 cases, 13 calls) | **12 (92%)** | **1 (8%)** | 0 |
| After fix `3e8be9a8…` (6 cases, 14 calls) | **11 (79%)** | **3 (21%)** | 0 |

**Takeaway (at that point):** earlier runs were almost only regulatory. After the Evidence/RETRIEVER work, academic showed up more but regulatory still dominated. Mix is mainly **agent tool-choice / prompt** behavior.

---

## Follow-up — explicit tool-choice in `SYSTEM_PROMPT`

Added prompt rules: always `search_regulatory` for US MRM/SR/OCC; also `search_academic` at least once when the deck covers methodology, validation metrics, fairness, or PD modeling; prefer specialized tools over `hybrid_search_mra`.

**Re-eval run:** `56158c4c65a04763b96b0636154a7297` (job run `899006725656816`)

### Tool mix across runs

| Eval run | `search_regulatory` | `search_academic` | hybrid |
|----------|---------------------|-------------------|--------|
| Baseline `17254066…` (13 calls) | **12 (92%)** | **1 (8%)** | 0 |
| Evidence/RETRIEVER `3e8be9a8…` (14 calls) | **11 (79%)** | **3 (21%)** | 0 |
| Tool-choice prompt `56158c4c…` (13 calls) | **8 (62%)** | **5 (38%)** | 0 |

After the tool-choice prompt, academic appeared on **all 6 cases**.

### Scorer metrics across runs

| Scorer | Baseline `17254066…` | Evidence fix `3e8be9a8…` | Tool-choice `56158c4c…` |
|--------|----------------------|--------------------------|-------------------------|
| `decision_label_check` | 1.0 | 1.0 | 1.0 |
| `report_structure_check` | 1.0 | 1.0 | 1.0 |
| `decision_policy` | 1.0 | 1.0 | 1.0 |
| `report_sections` | 1.0 | 1.0 | **0.83** |
| `grounded_evidence` | **0.0** | **0.17** | **0.33** |
| `retrieval_groundedness` | — | 0.05 | **0.0** |

**Takeaway:** tool-choice prompt clearly fixed the regulatory/academic imbalance. `grounded_evidence` kept improving; `report_sections` / `retrieval_groundedness` did not — remaining quality work is claim–chunk alignment and section completeness, not tool routing.
