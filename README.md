# Agency Shadow — V9 research validation

Synthetic research prototype for reactive oversight and transport sensitivity. No human participants, private records, or LLM inference. Transport bounds are externally assumed in this experiment; the misspecified scenario shows they cannot be treated as safety guarantees without evidence.

## Reproduce
Python 3.10+; CPU only.

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.experiment
python -m src.cost_design
python -m src.transport
# Optional independent optimization check:
pip install scipy
python -m scripts.lp_oracle
```

The three simulation modules produce JSON results in `results/`. Repeated and first-probe histories estimate different quantities; equal assumed costs do not establish real recruitment savings. The V9 transport experiment compares three approaches under both valid and deliberately violated bias-drift assumptions. The gate checks relative corrective-agency performance only, not overall model safety. Do not treat simulator outputs as human measurements.
