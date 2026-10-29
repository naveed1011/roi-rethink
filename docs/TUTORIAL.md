# Tutorial: run it, collect the results, publish it

This is the short version of everything I did to verify the project. It
assumes Windows + Anaconda (my setup), but the Python commands are the same
on macOS/Linux. Total time: about 10 minutes.

You will end up with:

- 23 passing tests,
- a synthetic dataset in `data/synthetic/`,
- an eval JSON with per-sample records in `results/`,
- three regenerated figures in `assets/`,
- a clean git tree ready to push.

---

## 1. Create / activate the environment

I use a dedicated env:

```powershell
conda create -n roi-rethink python=3.11 -y
conda activate roi-rethink
```

If you already have an env with the dependencies (mine is `learning_env`),
just activate that instead:

```powershell
conda activate learning_env
```

Check which Python you are actually using — this is the most common source of
"module not found" confusion:

```powershell
python -c "import sys; print(sys.executable)"
```

## 2. Install

From the project root (the folder containing `README.md`):

```powershell
pip install -r requirements.txt
pip install -e .
```

`pip install -e .` makes `import roi_rethink` work from anywhere in the env.
It is not strictly required for the scripts (they add the repo root to
`sys.path`), but it is how a user would consume the package.

## 3. Run the tests first

```powershell
pytest tests/ -v
```

Expected:

```text
23 passed
```

If this fails, stop here. Nothing downstream is meaningful.

## 4. Generate the synthetic dataset

```powershell
python scripts/generate_data.py --n 20 --out data/synthetic
```

Expected:

```text
Generated 20 samples -> data/synthetic
```

This writes `data/synthetic/images/sample_0000.png` … `sample_0019.png` and
`metadata.json` with the ground-truth box and anomaly kind for each sample.
`data/` is gitignored on purpose — generated data should not be pushed.

Open a few images to sanity-check that the planted blobs are visible.

## 5. Run the offline evaluation

```powershell
python scripts/run_eval.py --mock --n 10 --seed 7 --out results/demo.json
```

Expected shape (exact values can shift slightly across Pillow versions):

```json
{
  "n": 10,
  "mean_reward": 0.384,
  "mean_format": 1.0,
  "mean_iou": 0.015,
  "mean_iou_positive": 0.029,
  "valid_mark_rate": 1.0,
  "detection_accuracy": 0.5,
  "iou_at_0_5": 0.0,
  "anomaly_prevalence": 0.5
}
```

What to check:

- `valid_mark_rate` and `mean_format` should be **1.0**. If they are not, the
  loop is not running end-to-end.
- `mean_iou_positive` will be near zero. That is expected: `MockVLM` is blind
  and always emits the same box. It tests the plumbing, not intelligence.

Add `--quiet` if the progress bar messes up your terminal or log:

```powershell
python scripts/run_eval.py --mock --n 10 --seed 7 --out results/demo.json --quiet
```

`results/` is gitignored. To inspect one trajectory, open
`results/demo.json` and look at `records[0]`: it contains `turn1`, the marked
bbox, the final answer, and the reward breakdown.

## 6. Regenerate the figures

```powershell
python scripts/make_figures.py --out assets
```

Expected:

```text
wrote assets\synthetic_samples.png
wrote assets\loop_example.png
wrote assets\eval_mock.png
```

These are the exact images the README embeds, so the figures are reproducible
rather than hand-made. The loop figure uses a scripted GT box for
illustration; the metrics figure uses the blind mock and says so.

## 7. Try the interactive demo (optional)

```powershell
python scripts/demo_gradio.py
```

Open the printed URL (normally <http://127.0.0.1:7860>). Leave the image box
empty to use a generated scan, click **Run two-turn reasoning**, and you
should see:

- the marked ROI on the right,
- the turn-1 hypothesis and turn-2 answer below.

This is the screen to screenshot if you want an extra image for your
portfolio. Stop it with `Ctrl+C`.

## 8. Optional: a real VLM

Only do this if you have an API key and are willing to spend a little money:

```powershell
$env:OPENAI_API_KEY = "sk-..."
python scripts/run_eval.py --model gpt-4o-mini --n 20 --out results/eval_real.json
```

Check `valid_mark_rate` first. A real model often needs a retry to produce a
valid `<tool_call>`; the agent retries once. If the rate is low, fix the
prompt before interpreting IoU.

I have not run a paid API call in this repo, so any real-model numbers you
produce are yours — do not copy the mock numbers into a claim about a VLM.
