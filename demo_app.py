"""Small live RepairRL demo on the local GPU, reachable through SSH forwarding."""

import argparse
from dataclasses import dataclass
import gc
import json
import os
from pathlib import Path
import re
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
        self.cases = {
            "Cup swap": (
                "runbugrun",
                banks["runbugrun"]["21346"],
                "Try GRPO or DPO for a repair, then M1 SFT to show the recorded failure.",
            ),
            "Remove a debug print": (
                "runbugrun",
                banks["runbugrun"]["186967"],
                "Compare M1 SFT or GRPO with DPO; an extra print can break the output contract.",
            ),
            "Largest proper divisor": (
                "humanevalpack",
                banks["humanevalpack"]["Python/24"],
                "Compare pretrained with Instruct. DPO failed first attempt but recovered with feedback in the recorded evaluation.",
            ),
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
        statement = re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", row["statement"]).strip()
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
            return result["candidate"], verdict, observations, f"**Ready:** {label}"


def create_app(demo):
    import gradio as gr

    initial = next(iter(demo.cases))
    model = next(iter(demo.models))
    statement, code, hint, output, status, input_status, trace = demo.preview(initial)
    css = ".gradio-container {max-width: 1300px !important; margin: auto;} .code-panel {min-height: 320px;} .cm-editor {font-size: 14px !important;}"
    with gr.Blocks(
        title="RepairRL — live repair",
        theme=gr.themes.Soft(),
        css=css,
        analytics_enabled=False,
        fill_width=True,
    ) as app:
        gr.Markdown(
            "# RepairRL\nChoose a problem and a model. Edit the program, then ask the model to fix it."
        )
        with gr.Row():
            problem = gr.Dropdown(list(demo.cases), value=initial, label="Problem")
            model_choice = gr.Dropdown(list(demo.models), value=model, label="Model")
        ready = gr.Markdown(
            f"**Ready:** {demo.loaded}"
            if demo.loaded
            else "Model loads when selected or when you click Fix bug."
        )
        description = gr.Textbox(
            value=statement,
            label="Problem description",
            lines=5,
            max_lines=9,
            interactive=False,
            elem_id="problem-description",
        )
        guide = gr.Markdown(hint)
        with gr.Row():
            with gr.Column():
                editor = gr.Code(
                    value=code,
                    language="python",
                    label="Your program · editable",
                    lines=15,
                    max_lines=24,
                    interactive=True,
                    elem_classes=["code-panel"],
                    wrap_lines=True,
                )
                tested = gr.Markdown(input_status)
            with gr.Column():
                generated = gr.Code(
                    value=output,
                    language="python",
                    label="Model response",
                    lines=15,
                    max_lines=24,
                    interactive=False,
                    elem_classes=["code-panel"],
                    wrap_lines=True,
                )
                verdict = gr.Markdown(status)
        with gr.Row():
            fix = gr.Button("Fix bug", variant="primary")
            test = gr.Button("Test current code")
            reset = gr.Button("Reset problem")
        retry = gr.Checkbox(
            value=False, label="Retry with test feedback · up to 3 attempts"
        )
        with gr.Accordion("Attempt details", open=False):
            details = gr.JSON(value=trace)
        gr.Markdown(
            "Curated held-out examples. Buttons run live inference and real tests; edited inputs can change the outcome."
        )
        problem.change(
            demo.preview,
            problem,
            [description, editor, guide, generated, verdict, tested, details],
            api_name="select_problem",
            concurrency_id="demo",
            concurrency_limit=1,
        )
        reset.click(
            demo.preview,
            problem,
            [description, editor, guide, generated, verdict, tested, details],
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
            demo.fix,
            [problem, model_choice, editor, retry],
            [generated, verdict, details, ready],
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
