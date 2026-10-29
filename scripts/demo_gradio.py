#!/usr/bin/env python3
"""Interactive Gradio demo of the two-turn mark->rethink loop.

Run:  python scripts/demo_gradio.py
Then open the printed local URL. Upload any image (or use a generated
synthetic scan) and watch: turn-1 hypothesis + marked ROI, then turn-2
rethink + final answer, side by side.

Uses MockVLM by default (fully offline). Pass --model <name> to use a
real OpenAI-compatible VLM (requires OPENAI_API_KEY).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from roi_rethink.agent import RethinkAgent  # noqa: E402
from roi_rethink.synthetic import SyntheticAnomalyDataset  # noqa: E402
from roi_rethink.vlm import MockVLM, OpenAICompatibleVLM  # noqa: E402


def build_demo(agent: RethinkAgent):
    import gradio as gr

    ds = SyntheticAnomalyDataset(seed=123)

    def run(image, query):
        if image is None:
            sample = next(ds.iter(1, start_seed=999))
            image = sample.image
            query = query or sample.prompt
        traj = agent.run(image, query or "Assess this image for any anomaly.")
        marked = traj.marked_image or image
        summary = (
            f"**Turn 1 (hypothesis + mark):**\n```\n{traj.turn1}\n```\n\n"
            f"**Marked bbox:** {traj.marked_bbox}  (valid: {traj.valid_mark})\n\n"
            f"**Turn 2 (rethink + answer):**\n```\n{traj.turn2}\n```\n\n"
            f"**Final answer:** {traj.final_answer}"
        )
        return marked, summary

    with gr.Blocks(title="roi-rethink: two-turn grounded visual reasoning") as demo:
        gr.Markdown(
            "# roi-rethink\n"
            "A lightweight re-implementation of BrReMark's two-turn "
            "*hypothesis -> mark ROI -> rethink -> answer* loop. "
            "Upload an image (or leave blank for a synthetic scan), submit a query, "
            "and inspect the marked ROI and final grounded answer."
        )
        with gr.Row():
            with gr.Column():
                img_in = gr.Image(type="pil", label="Input image (optional)")
                q_in = gr.Textbox(
                    label="Query",
                    value="Assess this image for any suspected anomaly. If present, mark its location.",
                )
                btn = gr.Button("Run two-turn reasoning")
            with gr.Column():
                img_out = gr.Image(label="Turn-2 observation (marked ROI)")
        txt_out = gr.Markdown(label="Trajectory")
        btn.click(run, inputs=[img_in, q_in], outputs=[img_out, txt_out])
    return demo


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default=None, help="OpenAI-compatible model name; omit for offline MockVLM")
    p.add_argument("--share", action="store_true")
    args = p.parse_args()

    if args.model:
        vlm = OpenAICompatibleVLM(model=args.model)
    else:
        vlm = MockVLM()
    agent = RethinkAgent(vlm)
    demo = build_demo(agent)
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
