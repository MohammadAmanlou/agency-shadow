# Agency Shadow — V12 research validation

Synthetic research prototype for reactive oversight and transport sensitivity. No human participants, private records or LLM inference in the checked-in results. The sparse violation budget k and proposed edge radii are external assumptions, not learned safety guarantees. Use the direct mode when no defensible assumption exists. A sentinel can test a particular edge from independent confidence intervals; it cannot validate unmeasured edges.

## Reproduce (CPU, Python 3.10+)

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.experiment
python -m src.cost_design
python -m src.transport
python -m src.robust_audit
python -m src.cut_audit
python -m src.calibrated_gate
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

## V11: cut-witness design
The KL covering LP is an expected-cost lower bound for a known safe baseline and finite unsafe cut alternatives. The LR gate controls false passes for those listed alternatives only. The LP allocation is not a finite-sample optimum. `upper_bad_count` uses exact hypergeometric inversion for a frozen graph and uniform edge checks; noisy validation costs and the integration with a composite-alternative gate remain research work. Outside-class failures and the cycle baseline are retained in results.

## V12: joint calibration and the obstruction
The composite noninferiority gate does not take the true reference or violation count as input (except the diagnostic oracle). It combines finite-look simultaneous intervals and a frozen uniform edge sample; all samples are charged. The calibrated and targeted methods did not activate graph inference in the 960-run exploratory suite. Direct audit was cheaper at comparable performance. Keep this negative result: a graph is not automatically useful just because its constraints are plausible. No human or LLM validation, no global novelty claim. The oracle is not a deployable baseline. Full JSON regenerated locally; checked-in CSV summarizes the executed runs.
