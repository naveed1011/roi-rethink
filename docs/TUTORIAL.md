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

---

## 9. Clean up before committing

Remove local caches (already gitignored, but this keeps the folder tidy):

```powershell
Remove-Item -Recurse -Force .pytest_cache, roi_rethink.egg-info -ErrorAction SilentlyContinue
```

Keep `assets/` — those PNGs are tracked and referenced by the README. You can
keep `data/` and `results/` locally for your own inspection; they are
gitignored either way.

If you have stray screenshots or notes in the project root, delete or move
them outside the repo. They are not part of the project.

Check for secrets:

```powershell
git grep -n -i "sk-" -- . 2>$null
```

(If the repo is not initialized yet, use
`Select-String -Path README.md,*.py,scripts\*.py -Pattern "sk-"`.)

## 10. Push to GitHub

Create an empty repository on GitHub first (no README, no .gitignore, to
avoid a merge), then:

```powershell
git init
git add .
git status
```

**Read the `git status` output before committing.** It should list source
files, tests, docs, assets, `README.md`, `LICENSE`, `.gitignore`,
`requirements.txt`, `pyproject.toml`, and `.github/workflows/ci.yml`. It
should **not** list `data/`, `results/`, `__pycache__/`, `.pytest_cache/`,
`roi_rethink.egg-info/`, or `.env`.

Then:

```powershell
git config --global user.name "Naveed Ahmad"
git config --global user.email "you@example.com"

git commit -m "roi-rethink: two-turn VLM mark-and-rethink loop for medical ROI anomaly detection"
git branch -M main
git remote add origin https://github.com/<your-username>/roi-rethink.git
git push -u origin main
```

If GitHub asks for a password, use a personal access token, not your account
password. If the remote already has a commit (e.g. you let GitHub create a
README), pull first:

```powershell
git pull --rebase origin main
git push -u origin main
```

After the push, check the **Actions** tab: CI should run the 23 tests on
Python 3.9 and 3.11. If it fails there but passes locally, it is almost always
a missing optional dependency — the tests skip `gradio`/`openai` when they are
not installed.

## 11. Repo settings I would use

- **Name:** `roi-rethink`
- **Description:** `Two-turn mark-and-rethink VLM loop for grounded medical ROI anomaly detection — an educational re-implementation of BrReMark that runs offline.`
- **Topics:** `medical-imaging`, `vlm`, `reinforcement-learning`, `grpo`,
  `anomaly-detection`, `brain-mri`, `python`, `gradio`,
  `vision-language-model`

---

## Troubleshooting

**`python` is the wrong interpreter**
`where python` should point into your env. If not, `conda activate` it again,
or run `conda run -n roi-rethink python scripts/run_eval.py ...`.

**`ModuleNotFoundError: No module named 'roi_rethink'`**
Run from the repo root, or `pip install -e .`.

**`No module named pytest` / `gradio` / `matplotlib`**
`pip install -r requirements.txt`.

**PowerShell prints `NativeCommandError` during eval**
The `tqdm` progress bar writes to stderr, and piping stderr in PowerShell can
surface it as an error even when the exit code is 0. Use `--quiet`, or run the
command without `2>&1 | ...`.

**Gradio port already in use**
Close the other demo, or set `$env:GRADIO_SERVER_PORT = "7861"` before
running it.

**`OPENAI_API_KEY` missing**
Expected if you use `--model`; the `--mock` path needs no key.

**Figures look different from the README**
Pillow's Gaussian blur can differ slightly across versions. The structure and
the story stay the same; regenerate with the installed version.

---

## Pre-push checklist

- [ ] `pytest tests/ -v` → 23 passed
- [ ] `python scripts/run_eval.py --mock --n 10 --seed 7 --quiet` → format 1.0, valid mark rate 1.0
- [ ] `python scripts/make_figures.py --out assets` → three PNGs updated
- [ ] `python examples/quickstart.py` → `total=1.000` (harness sanity check)
- [ ] README images render locally (relative paths under `assets/`)
- [ ] `git status` has no `data/`, `results/`, caches, egg-info, screenshots, or `.env`
- [ ] no API keys in tracked files
- [ ] CI green after the push

If all of that is true, you can push with confidence. The code is verified;
what the repo honestly does *not* claim is a clinical result from a real VLM.
