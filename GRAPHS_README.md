# CS725 Graph Generator — Quick Reference

Run from the folder containing `run_demo.py`:

```bash
python -m pip install -r requirements.txt
python run_demo.py --dataset synthetic
python generate_graphs.py
```

Open `graphs/index.html` in any browser. The script saves **8 charts**, each in
**PNG** and **SVG** by default, along with chart metadata and summary CSVs.
The graphs always read the saved experiment data; there are no hard-coded results.

## To use real Adult data

Place the official `adult.data` and `adult.test` files in `data/`, then:

```bash
python run_demo.py --dataset adult
python generate_graphs.py
```

## Optional commands

```bash
python generate_graphs.py --format png
python generate_graphs.py --results-dir ./outputs --output-dir ./my_charts --dpi 300
python generate_graphs.py --help
```

## Inputs

Required: `outputs/classifier_metrics.json`, `outputs/comparison_summary.csv`,
`outputs/per_query_results.csv`.

Optional: `outputs/run_metadata.json`, `outputs/query_examples.json`,
`outputs/demo_example.json`.

## Important limitations

- The bundled charts show *synthetic Adult-like* data. They must not be labelled as UCI Adult findings.
- The starter generators have unequal search spaces and budgets, so you cannot interpret the chart differences as a causal effect of adding constraints.
- Cost scores are arbitrary effort units; they are not money.
- A 'PASS' verdict is a pass under *the project's declared validator rules only*, not verified real-world feasibility.
- A 'no returned flip' result may mean that this limited candidate search failed; it does **not** prove that no counterfactual exists.
- The recorded numeric classifier probabilities are model scores, not guarantees of future income.
