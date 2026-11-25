# Agency Shadow

Research prototype for reactive oversight evaluation. Synthetic experiments only: no human participants, private records, or LLM inference. No established novelty or empirical safety claim.

## Reproduce
Python 3.10+; CPU only.

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.experiment --seed 20261125 --n 1200 --horizon 12 --trials 300
python -m src.cost_design
```

Repeated probes and first probes estimate different quantities. Equal probe budgets need not imply equal recruitment costs. The cost-design module illustrates bounded-bias sampling with declared sensitivity assumptions, not a validated simulator of human behavior. The research proposal is maintained separately for applicant and supervisor review.
