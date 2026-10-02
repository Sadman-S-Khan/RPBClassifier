# Release checklist for rpboost (NOT RELEASED YET)

Status: **prepared, not finalized.** Per the author's decision, nothing is published until the 31-dataset real-data benchmark (`../New_Algorithm/real_benchmark.py`) has finished and any changes it motivates are made. The version on PyPI must be that final, updated version.

## Current state (2 Oct 2026): version 1.2.0, ready but NOT released
- [x] Benchmark finished; the result-driven change (categorical handling) was made as RPB v1.2 and re-benchmarked with the same protocol (`../New_Algorithm/Benchmark_v1.2_Results.md`)
- [x] `src/rpboost/_rpb.py` algorithm code is byte-identical to `../New_Algorithm/rpb.py` v1.2; predictions match exactly (verified on 4 datasets, rules on/off)
- [x] 64 tests pass (sklearn `check_estimator` + package tests, including categorical/rules tests)
- [x] README benchmark section filled in; version 1.2.0 in `__init__.py` and `CITATION.cff`

## Earlier state (1 Oct 2026)
- [x] Package layout (`src/rpboost`), `pyproject.toml`, BSD-3-Clause `LICENSE`, `CITATION.cff`, `README.md`, `examples/quickstart.py`, CI workflow (`.github/workflows/tests.yml`)
- [x] Passes scikit-learn's full `check_estimator` suite plus the package's own tests (62 passed)
- [x] Builds cleanly (`uv build`), `twine check` passes, and it installs and runs in a clean environment
- [x] The algorithm is identical to `../New_Algorithm/rpb.py`, the version being benchmarked. The only additions are API polish: `decision_function`, `feature_names_in_`, clearer errors and scikit-learn-standard validation.

## Before release
1. **Benchmark finished** → review `../New_Algorithm/real_results.md`.
2. **Changes motivated by the results**, if any. The likely candidate is better handling of categorical / one-hot features, RPB's clearest weakness so far. Any algorithm change must be:
   - made in `src/rpboost/_rpb.py`;
   - re-benchmarked with the same protocol, so the paper and the package describe the same algorithm;
   - reflected in the version number (1.0.0 if unchanged at release; bump it if the algorithm changes after the paper's numbers are frozen).
3. **README:** replace "Benchmark status" with the final results table.
4. **GitHub repository:** https://github.com/Sadman-S-Khan/RPBClassifier (created 2 Oct 2026); `[project.urls]` filled in. Push the code and check that CI is green.
5. `CITATION.cff`: update the date; add `preferred-citation` once the paper (or its preprint) exists.

## Publishing (the author runs these; they need a PyPI account)
```bash
cd rpboost
rm -rf dist && uv build          # or: python -m build
uvx twine check dist/*
# 1) rehearse on TestPyPI (account at test.pypi.org, API token)
uvx twine upload --repository testpypi dist/*
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple rpboost
# 2) real release (account at pypi.org, API token)
uvx twine upload dist/*
```
After that, anyone can run `pip install rpboost` and then `from rpboost import RPBClassifier`.

Notes:
- A version number can be uploaded to PyPI only once and can never be reused, which is why we don't publish before the final version.
- The name `rpboost` was free on PyPI on 1 Oct 2026. It is only reserved once the first upload happens.
