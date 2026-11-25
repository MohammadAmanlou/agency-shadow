# Agency Shadow — V10 research validation

Synthetic research prototype for reactive oversight and transport sensitivity. No human participants, private records or LLM inference in the checked-in results. The sparse violation budget k and proposed edge radii are external assumptions, not learned safety guarantees. Use the direct mode when no defensible assumption exists. A sentinel can test a particular edge from independent confidence intervals; it cannot validate unmeasured edges.

## Reproduce (CPU, Python 3.10+)

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.experiment
python -m src.cost_design
python -m src.transport
python -m src.robust_audit
python -m src.benchmark
python -m scripts.lp_oracle
```

V10 adds a lifted difference-constraint graph (exact contrast bounds even for interval proxy means), deletion-set robustness, finite-look adaptive first-probe audits, and 480 synthetic fee-dispute cases with a deterministic evidence verifier. The experimental cost is hypothetical. The benchmark is a rule-engine unit benchmark, not evidence of human agency. The graph algorithm enumerates deletion sets and is intended for small version graphs.

## Optional LLM run
Start your local OpenAI-compatible inference server with a supported instruction model, then run:

```bash
python -m src.benchmark --endpoint http://127.0.0.1:8000/v1/chat/completions --model YOUR_MODEL --out results/llm_run
```

The endpoint receives synthetic rules and records, not labels or pair IDs. The runner tests factual resolution and evidence citations, not human behavior or learned agency. LLM run not yet performed. Do not use the gate to authorize deployment; a relative agency criterion is only one component of evaluation.
