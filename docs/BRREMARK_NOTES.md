# What I learned from BrReMark (and re-implemented here)

This project is an educational re-implementation of the core *mechanism*
of **BrReMark** (Li et al., 2026, Fudan University):
*Enhancing Brain MRI Anomaly Detection and Reasoning with ROI Rethink and
Synthetic Data*.

I did **not** re-train Lingshu-7B on 3,696 clinical MRI cases — that needs
multi-GPU nodes and licensed medical data. Instead, I isolated the four
ideas that make BrReMark interesting and built a small, runnable system
around each one so I could study and measure them directly.

## 1. The two-turn "hypothesis → mark → rethink → answer" loop

BrReMark's key insight is that a VLM gives better grounded diagnoses when
it is forced to *externally mark* a region of interest and then *re-examine
the marked image* in a second turn, instead of answering in one shot.

- **In BrReMark:** the model emits `<think>` → `<tool_call>mark_bbox` → the
  environment returns the image with the box drawn → the model emits
  `<rethink>` → `<answer>`.
- **Here:** `roi_rethink/agent.py` + `roi_rethink/env.py` enforce exactly
  this loop at inference time on *any* VLM via prompting, so you can run it
  without training. `env.py` is the `mark_bbox` environment; `agent.py`
  drives the two turns.

## 2. Composite reward for GRPO

BrReMark trains with GRPO (a variant of PPO) using a weighted reward:

| Term | Weight role | Implemented here |
|---|---|---|
| Format compliance (correct tags + valid tool call) | structure | `reward._format_score` |
| IoU localization (marked box vs GT lesion) | grounding | `reward.bbox_iou` |
| LLM-as-judge (clinical quality, 0–10) | correctness | pluggable `judge_fn` |

- **Here:** `reward.composite_reward` computes the same weighted sum. The
  LLM-judge term is an optional callback; when omitted it is *masked* and
  weights renormalize (see point 3).

## 3. Synthetic data with masked judge (Phase 2)

BrReMark does a clever two-phase RL: Phase 1 on real clinical data with the
full reward; Phase 2 continues on **SynthSeg-generated synthetic MRIs** with
the LLM-judge term **masked**, because synthetic scans have no clinical
ground truth to judge against — you'd otherwise reward hallucinated
"clinical correctness."

- **Here:** `synthetic.py` generates unlimited synthetic "scans" with a
  planted anomaly of known location/kind (our stand-in for SynthSeg).
  `run_eval.py` defaults to `judge_fn=None`, i.e. the judge is masked —
  exactly Phase-2 style.

## 4. Evaluation as LLM-as-judge + deterministic metrics

BrReMark's `eval/` uses an LLM judge with multi-dimensional 0–10 scoring.
- **Here:** `eval.py` keeps the deterministic metrics (format, IoU,
  IoU@0.5, detection rate) first-class and lets you plug in any judge, so
  results are reproducible without API calls.

## Mapping table

| BrReMark concept | This project | File |
|---|---|---|
| Two-turn reasoning trajectory | `RethinkAgent.run` | `roi_rethink/agent.py` |
| `mark_bbox` RL environment | `TwoTurnMarkEnv` | `roi_rethink/env.py` |
| Composite reward (format + IoU + judge) | `composite_reward` | `roi_rethink/reward.py` |
| SynthSeg synthetic data | `SyntheticAnomalyDataset` | `roi_rethink/synthetic.py` |
| vLLM two-turn inference | any OpenAI-compatible VLM + `MockVLM` | `roi_rethink/vlm.py` |
| `eval/run_judge.py` | `eval.evaluate` + `scripts/run_eval.py` | `roi_rethink/eval.py` |
| Lingshu-7B / VeRL / GRPO training | *not included* (out of scope; see "Next steps") | — |

## Next steps (if you want to go deeper)

1. **Ablation:** compare one-shot vs two-turn on the same VLM and same
   synthetic set; measure IoU lift. This is a real, publishable-lite
   experiment you can run for ~$5 of API calls.
2. **Plug in a real judge:** implement `judge_fn` with an LLM call that
   scores the final answer against `anomaly_kind`, and re-run eval with the
   full Phase-1 reward.
3. **Fine-tune a small VLM** (e.g. a 1–3B open VLM on a single GPU) on
   synthetic two-turn trajectories with the reward above — that is the
   full BrReMark pipeline at small scale. `docs/` has the data schema to
   get you started.
4. **Swap the domain:** replace synthetic scans with chest X-rays
   (NIH ChestX-ray14 is public) — same loop, real medical domain.
