"""Small live RepairRL demo on the local GPU, reachable through SSH forwarding."""

import argparse
from dataclasses import dataclass
import gc
import difflib
import html
import json
import os
from pathlib import Path
import threading
import time

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ModelSpec:
    model: str
    adapter: str | None
    revision: str


class RepairDemo:
    def __init__(self):
        from repairrl.common import BASE_REVISION, read_jsonl
        from agent.sandbox import Sandbox

        self.lock = threading.Lock()
        self.box = Sandbox(timeout=3.0)
        self.box.check()
        self.suites = json.loads(
            (ROOT / "data/prepared/scale-verified/tests.json").read_text()
        )
        banks = {
            b: {str(r["id"]): r for r in read_jsonl(ROOT / source)}
            for b, source in [
                ("runbugrun", "data/prepared/scale-verified/test.jsonl"),
                ("humanevalpack", "data/external/humanevalpack.jsonl"),
            ]
        }
        from demo_problems import PROBLEMS

        self.cases = {
            name: (
                meta["benchmark"],
                banks[meta["benchmark"]][meta["id"]],
                meta["hint"],
            )
            for name, meta in PROBLEMS.items()
        }
        parent = ROOT / "checkpoints/M1-merged-round2"
        specs = {
            "GRPO · step 800": ModelSpec(
                str(parent),
                str(ROOT / "results/round2/grpo-interim/step-800"),
                BASE_REVISION,
            ),
            "DPO · step 66": ModelSpec(
                str(parent),
                str(ROOT / "results/round2/remote/dpo-lr3e6/snapshots/step-66"),
                BASE_REVISION,
            ),
            "M1 · SFT": ModelSpec(str(parent), None, BASE_REVISION),
            "M0 · Instruct": ModelSpec(
                "Qwen/Qwen2.5-Coder-1.5B-Instruct", None, BASE_REVISION
            ),
            "Pretrained · no instruction tuning": ModelSpec(
                "Qwen/Qwen2.5-Coder-1.5B",
                None,
                "df3ce67c0e24480f20468b6ef2894622d69eb73b",
            ),
        }
        self.models = {}
        for label, spec in specs.items():
            if (
                spec.adapter
                and not (Path(spec.adapter) / "adapter_model.safetensors").exists()
            ):
                continue
            if (
                not spec.model.startswith("Qwen/")
                and not (Path(spec.model) / "config.json").exists()
            ):
                continue
            self.models[label] = spec
        if not self.models:
            raise RuntimeError("Restore the model artifacts before starting the demo.")
        self.loaded = None
        self.generate = None
        self.log = ROOT / "results/demo/events.jsonl"
        self.log.parent.mkdir(parents=True, exist_ok=True)

    def case(self, name):
        if name not in self.cases:
            raise ValueError("Choose one of the demo problems.")
        return self.cases[name]

    def preview(self, name):
        _, row, hint = self.case(name)
        from demo_problems import PROBLEMS

        statement = PROBLEMS[name]["description"]
        return (
            statement,
            row["buggy_code"],
            hint,
            "",
            "Choose a model, edit the code if needed, then click **Fix bug**.",
            "Current code has not been tested.",
            None,
        )

    def _load(self, label):
        if label not in self.models:
            raise ValueError("Unknown model.")
        if self.loaded == label:
            return 0.0
        import torch
        from repairrl.model import Generator
        from huggingface_hub import snapshot_download

        self.generate = None
        self.loaded = None
        gc.collect()
        torch.cuda.empty_cache()
        spec = self.models[label]
        model = spec.model
        if model.startswith("Qwen/"):
            model = snapshot_download(
                model, revision=spec.revision, local_files_only=True
            )
        start = time.monotonic()
        self.generate = Generator(
            model,
            spec.adapter,
            spec.revision,
            max_new_tokens=768,
            max_context=4096,
            seed=42,
            quantized=False,
        )
        self.loaded = label
        return time.monotonic() - start

    def load(self, label):
        with self.lock:
            elapsed = self._load(label)
            return f"**Ready:** {label}" + (
                f" · loaded in {elapsed:.1f}s" if elapsed else ""
            )

    def grade(self, name, raw):
        benchmark, row, _ = self.case(name)
        if benchmark == "runbugrun":
            from evaluation.runner import grade_program

            return grade_program(
                raw, self.suites[str(row["problem_id"])], row["problem_id"], self.box
            )
        from evaluation.humanevalpack_adapter import grade_humaneval

        return grade_humaneval(raw, row, self.box)

    def test_code(self, name, code):
        if not code.strip():
            raise ValueError("The editor is empty.")
        with self.lock:
            grade = self.grade(name, code)
            unit = "task suite" if self.case(name)[0] == "humanevalpack" else "tests"
            return f"**{'PASS' if grade.success else 'FAIL'}** · {grade.passed}/{grade.total} {unit} · `{grade.category}`"

    def fix(self, name, label, code, retry):
        if not code.strip():
            raise ValueError("The editor is empty.")
        if len(code.encode()) > 131072:
            raise ValueError("Program is too large.")
        from agent.repair import repair_task

        with self.lock:
            loaded = self._load(label)
            _, row, _ = self.case(name)
            task = row | {"buggy_code": code}
            start = time.monotonic()
            result = repair_task(
                task,
                self.generate,
                lambda raw: self.grade(name, raw),
                max_attempts=3 if retry else 1,
            )
            elapsed = time.monotonic() - start
            last = result["trace"][-1]["grade"]
            observations = [
                {
                    "attempt": t["attempt"],
                    "passed": t["grade"]["passed"],
                    "total": t["grade"]["total"],
                    "result": t["grade"]["category"],
                    "generation_seconds": round(t["generation"]["seconds"], 2),
                    "tokens": t["generation"]["tokens"],
                }
                for t in result["trace"]
            ]
            unit = "task suite" if self.case(name)[0] == "humanevalpack" else "tests"
            verdict = f"### {'PASS' if result['agent_at_3'] else 'FAIL'} · {last['passed']}/{last['total']} {unit}\n{label} · {result['attempts']} attempt(s) · {elapsed:.1f}s"
            if not result["agent_at_3"]:
                verdict += f" · `{last['category']}`"
            record = {
                "timestamp": time.time(),
                "problem": name,
                "model": label,
                "edited": code != row["buggy_code"],
                "load_seconds": loaded,
                "wall_seconds": elapsed,
                "pass_at_1": result["pass_at_1"],
                "agent_at_3": result["agent_at_3"],
                "input_code": code,
                "result": result,
            }
            with self.log.open("a") as log:
                log.write(json.dumps(record) + "\n")
            return (
                result["trace"][-1]["generation"]["text"],
                verdict,
                observations,
                f"**Ready:** {label}",
            )


def render_changes(before, response):
    """Escape all model text; show line changes without another model call."""
    from evaluation.runner import parse_program

    try:
        after = parse_program(response)
    except ValueError:
        return '<div class="diff-empty">The response is not a valid Python program. Open <b>Full response</b> to inspect it.</div>'
    left, right = before.splitlines(), after.splitlines()
    if left == right:
        return '<div class="diff-empty">No code changes. Check the test verdict below.</div>'
    rows = []
    added = removed = 0
    matcher = difflib.SequenceMatcher(None, left, right, autojunk=False)
    for group_index, group in enumerate(matcher.get_grouped_opcodes(3)):
        if group_index:
            rows.append(
                '<tr class="diff-gap"><td colspan="4">… unchanged lines …</td></tr>'
            )
        for tag, a, b, c, d in group:
            if tag == "equal":
                for old, new in zip(range(a, b), range(c, d)):
                    rows.append(
                        f"<tr><td>{old + 1}</td><td>{new + 1}</td><td> </td><td><pre>{html.escape(left[old])}</pre></td></tr>"
                    )
            if tag in ("replace", "delete"):
                for old in range(a, b):
                    removed += 1
                    rows.append(
                        f'<tr class="diff-removed"><td>{old + 1}</td><td></td><td>−</td><td><pre>{html.escape(left[old])}</pre></td></tr>'
                    )
            if tag in ("replace", "insert"):
                for new in range(c, d):
                    added += 1
                    rows.append(
                        f'<tr class="diff-added"><td></td><td>{new + 1}</td><td>+</td><td><pre>{html.escape(right[new])}</pre></td></tr>'
                    )
    return f'<div class="diff-summary">+{added} added · −{removed} removed <span>Old / new line numbers</span></div><div class="diff-scroll"><table class="code-diff"><tbody>{"".join(rows)}</tbody></table></div>'


def create_app(demo):
    import gradio as gr

    initial = next(iter(demo.cases))
    model = next(iter(demo.models))
    statement, code, hint, output, status, input_status, trace = demo.preview(initial)
    css = """
    .gradio-container {max-width: 1600px !important; margin: auto;}
    .problem-pane.prose {max-height: 73vh; overflow-y: auto; padding: 8px 18px 18px; border: 1px solid #dce3eb; border-radius: 12px; background: white;}
    .problem-pane .prose {font-size: 16px; line-height: 1.65; max-width: none;}
    .problem-pane pre {font-size: 14px;}
    .problem-pane h2 {margin-top: 10px;}
    .cm-editor {font-size: 15px !important;}
    .code-panel {min-height: 490px;}
    #action-bar {gap: 8px; align-items: center;}
    #action-bar button {flex-grow: 0 !important; min-height: 34px; padding: 5px 14px;}
    .diff-summary {padding: 12px; font-weight: 600; color: #24533c;}
    .diff-summary span {float:right; font-weight:400; color:#64748b; font-size:12px;}
    .diff-scroll {max-height: 62vh; overflow: auto; border:1px solid #dce3eb; border-radius:8px;}
    .code-diff {width:100%; border-collapse:collapse; font-family:monospace; font-size:14px;}
    .code-diff td {padding:3px 6px; vertical-align:top; border:0;}
    .code-diff td:nth-child(1), .code-diff td:nth-child(2) {width:32px; text-align:right; color:#64748b; user-select:none;}
    .code-diff td:nth-child(3) {width:15px;}
    .code-diff pre {margin:0; white-space:pre; font:inherit; background:transparent; color:inherit;}
    .diff-added {background:#dcfce7; color:#14532d;}
    .diff-removed {background:#fee2e2; color:#7f1d1d;}
    .diff-gap,.diff-empty {padding:20px; color:#64748b;}
    """
    empty_diff = (
        '<div class="diff-empty">Changed lines will appear here after Fix bug.</div>'
    )
    with gr.Blocks(
        title="RepairRL — live repair",
        theme=gr.themes.Soft(),
        css=css,
        analytics_enabled=False,
        fill_width=True,
    ) as app:
        gr.Markdown("# RepairRL · live code repair")
        with gr.Row():
            problem = gr.Dropdown(
                list(demo.cases), value=initial, label="Problem", scale=3
            )
            model_choice = gr.Dropdown(
                list(demo.models), value=model, label="Model", scale=2
            )
        ready = gr.Markdown(
            f"**Ready:** {demo.loaded}" if demo.loaded else "Choose a model to begin."
        )
        with gr.Row():
            with gr.Column(scale=4, min_width=340):
                description = gr.Markdown(
                    statement,
                    elem_classes=["problem-pane"],
                    latex_delimiters=[
                        {"left": "$$", "right": "$$", "display": True},
                        {"left": "$", "right": "$", "display": False},
                    ],
                )
            with gr.Column(scale=6, min_width=520):
                with gr.Row(elem_id="action-bar"):
                    fix = gr.Button(
                        "Fix bug", variant="primary", size="sm", scale=0, min_width=100
                    )
                    test = gr.Button("Test code", size="sm", scale=0, min_width=100)
                    reset = gr.Button("Reset", size="sm", scale=0, min_width=80)
                retry = gr.Checkbox(
                    value=False, label="Use feedback · up to 3 attempts"
                )
                with gr.Tabs(selected="editor") as tabs:
                    with gr.Tab("Editor", id="editor"):
                        editor = gr.Code(
                            value=code,
                            language="python",
                            label="Your program · editable",
                            lines=23,
                            max_lines=30,
                            interactive=True,
                            wrap_lines=True,
                            elem_classes=["code-panel"],
                        )
                    with gr.Tab("Changed lines", id="changes"):
                        changed = gr.HTML(empty_diff)
                    with gr.Tab("Full response", id="response"):
                        generated = gr.Code(
                            value=output,
                            language="python",
                            label="Model response",
                            lines=23,
                            max_lines=30,
                            interactive=False,
                            wrap_lines=True,
                            elem_classes=["code-panel"],
                        )
                tested = gr.Markdown(input_status)
                verdict = gr.Markdown(status)
        with gr.Accordion("Presenter notes / attempt details", open=False):
            guide = gr.Markdown(hint)
            details = gr.JSON(value=trace)
        gr.Markdown(
            "Curated held-out cases · live inference and tests · highlighted lines show edits, not an explanation of the model’s reasoning."
        )

        def preview(name):
            return (*demo.preview(name), empty_diff, gr.update(selected="editor"))

        def fix_with_changes(name, label, submitted, retry):
            candidate, verdict, details, ready = demo.fix(name, label, submitted, retry)
            # Display the actual last raw response in its own tab.
            raw = candidate
            return (
                candidate,
                verdict,
                details,
                ready,
                render_changes(submitted, raw),
                gr.update(selected="changes"),
            )

        outputs = [
            description,
            editor,
            guide,
            generated,
            verdict,
            tested,
            details,
            changed,
            tabs,
        ]
        problem.change(
            preview,
            problem,
            outputs,
            api_name="select_problem",
            concurrency_id="demo",
            concurrency_limit=1,
        )
        reset.click(
            preview,
            problem,
            outputs,
            api_name="reset_problem",
            concurrency_id="demo",
            concurrency_limit=1,
        )
        model_choice.change(
            demo.load,
            model_choice,
            ready,
            api_name="select_model",
            concurrency_id="demo",
            concurrency_limit=1,
        )
        test.click(
            demo.test_code,
            [problem, editor],
            tested,
            api_name="test_code",
            concurrency_id="demo",
            concurrency_limit=1,
        )
        fix.click(
            fix_with_changes,
            [problem, model_choice, editor, retry],
            [generated, verdict, details, ready, changed, tabs],
            api_name="fix_bug",
            concurrency_id="demo",
            concurrency_limit=1,
        )
    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--no-preload", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    import torch

    torch.set_num_threads(4)
    demo = RepairDemo()
    if not args.no_preload:
        print(demo.load(next(iter(demo.models))), flush=True)
    create_app(demo).queue(max_size=16).launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=False,
        inbrowser=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
