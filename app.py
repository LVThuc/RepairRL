"""Local held-out repair demo using the same grader and three attempts."""
import argparse
import gc
import json
from pathlib import Path
import threading
from agent.repair import repair_task
from agent.sandbox import Sandbox
from evaluation.runner import grade_program
from repairrl.common import read_jsonl


def create_app(data="data/prepared/verified/test.jsonl", tests="data/prepared/verified/tests.json", checkpoint_root="checkpoints"):
    import gradio as gr
    tasks = {str(r["id"]): r for r in read_jsonl(data)}
    suites = json.loads(Path(tests).read_text())
    box = Sandbox()
    box.check()
    models = {"M0 Base": ("Qwen/Qwen2.5-Coder-1.5B-Instruct", None)}
    for arm, label in [("M1", "SFT"), ("M2", "SFT + weak-negative DPO"), ("M3", "SFT + execution GRPO")]:
        path = Path(checkpoint_root) / arm
        if (path / "adapter_config.json").exists():
            metadata = json.loads((path / "repairrl-training.json").read_text())
            models[f"{arm} {label}"] = (metadata["config"]["model"], str(path))
    cache, lock = {}, threading.Lock()
    def preview(task_id):
        r = tasks[task_id]
        return r["statement"], r["buggy_code"]
    def run(task_id, checkpoint):
        with lock:
            if cache.get("key") != checkpoint:
                cache.clear()
                gc.collect()
                import torch
                torch.cuda.empty_cache()
                from repairrl.model import Generator
                model, adapter = models[checkpoint]
                cache["generate"] = Generator(model, adapter)
                cache["key"] = checkpoint
            task = tasks[task_id]
            result = repair_task(task, cache["generate"], lambda raw: grade_program(
                raw, suites[str(task["problem_id"])], task["problem_id"], box))
            observations = [{"attempt": t["attempt"], "passed": t["grade"]["passed"],
                             "total": t["grade"]["total"], "result": t["grade"]["category"],
                             "generation_seconds": t["generation"]["seconds"]} for t in result["trace"]]
            return result["candidate"], {"success": result["agent_at_3"], "attempts": observations}
    with gr.Blocks(title="RepairRL") as app:
        gr.Markdown("# RepairRL\nPython repair with up to three feedback-guided attempts. Available checkpoints only.\n"
                    f"Checkpoints: `{checkpoint_root}`. Cases come from the frozen held-out split; an individual demo is not aggregate evidence.")
        task = gr.Dropdown(list(tasks), value=next(iter(tasks)), label="Held-out task")
        model = gr.Dropdown(list(models), value="M0 Base", label="Checkpoint")
        statement = gr.Textbox(label="Problem", lines=8)
        buggy = gr.Code(label="Buggy program", language="python")
        button = gr.Button("Repair and test")
        output = gr.Code(label="Latest candidate", language="python")
        trace = gr.JSON(label="Execution summary")
        task.change(preview, task, [statement, buggy])
        app.load(preview, task, [statement, buggy])
        button.click(run, [task, model], [output, trace], concurrency_limit=1)
    return app


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/prepared/verified/test.jsonl")
    p.add_argument("--tests", default="data/prepared/verified/tests.json")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--checkpoint-root", default="checkpoints")
    a = p.parse_args()
    create_app(a.data, a.tests, a.checkpoint_root).queue().launch(server_name="127.0.0.1", server_port=a.port, share=False)
