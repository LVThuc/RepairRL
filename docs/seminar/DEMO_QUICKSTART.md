# Live demo on the local RTX3060

`demo_app.py` runs on the machine with the3060. It binds to127.0.0.1:7860 and is accessed from another computer through SSH local forwarding. Inference and test execution both stay on the3060 computer; neither the browser computer nor the rented5070 server is required for model execution.

## Open it remotely

On the other computer, establish SSH access to the3060 machine, then run:

```bash
ssh -N -L 7860:127.0.0.1:7860 USER@3060_HOST
```

Keep that terminal open and visit **http://localhost:7860** in the browser on the other computer. `3060_HOST` is the SSH-accessible3060 machine, not the rental server. If using its Tailscale address, the browser computer must be connected to the same tailnet and permitted by its SSH policy. Ordinary LAN SSH requires its own running SSH service. Configure and verify SSH access from the browser computer before presenting.

If local port7860 is occupied on the browser computer, use `-L 7861:127.0.0.1:7860` and open http://localhost:7861.

## Start/stop on the GPU computer

On the original configured workstation, a user service was installed:

```bash
systemctl --user status repairrl-demo
systemctl --user restart repairrl-demo
journalctl --user -u repairrl-demo -f
systemctl --user stop repairrl-demo
```

That workstation service is not enabled at boot. It is not installed automatically by cloning this repository. On that workstation, start it with `systemctl --user start repairrl-demo`. This computer needs to stay on while presenting.

Portable manual launch from the project root:

```bash
.venv/bin/python demo_app.py --port 7860
```

Default behavior preloads GRPO step800. Use `--no-preload` to load on the first selection/fix instead. The app keeps one model in GPU memory at a time; changing models replaces it. It uses FP16, greedy generation, seed42, context4096 and768 completion tokens, matching the current V1 evaluation settings. No explicit thinking traces or separate diagnosis calls are used.

## Interface

1. Select one of three curated held-out problems.
2. Read the statement and edit the prefilled program in the left code editor.
3. Choose pretrained, Instruct, SFT, DPO step66 or GRPO step800.
4. Click **Test current code** to score the edited input, or **Fix bug** for live repair.
5. The right panel displays the actual generated program and its test verdict.
6. Optional: enable **Retry with test feedback** for at most3 V1 attempts. Default is one attempt so model failures remain visible.

**Reset problem** restores the original buggy program. Editing the code changes what the model receives and what the test button grades. It does not change the trusted problem tests. Generated responses are not replaced with cached/gold fixes. The largest-divisor task has a binary whole-suite verdict; it does not report one independently scored assertion as if it were all assertions.

## Suggested five-minute sequence

| Step | Problem/model | Verified live outcome |
|---|---|---|
|1|Cup swap / M1 SFT|Fail,1/3 selected tests|
|2|Cup swap / GRPO step800|Pass,3/3|
|3|Cup swap / DPO step66|Pass,3/3|
|4|Debug print / DPO step66|Fail,0/10|
|5|Debug print / GRPO step800|Pass,10/10|
|6|Largest divisor / pretrained|Invalid output format|
|7|Largest divisor / M0 Instruct|Pass|
|8|Largest divisor / DPO step66, feedback enabled|First attempt fails; second passes|

These are explicitly selected demonstrations. Outcomes were reproduced live during implementation, but edits, software changes or numerical differences can change outputs. Do not describe selected cases as representative aggregate performance.

## Measured live latency

After model loading, observed GRPO repair+test time was about4.1sec for cup-swap,10.0sec for debug-print and1.6sec for largest-divisor. Initial GRPO loading took5.9sec; switches also add load time. Timings vary with GPU load.

## Artifacts and requirements

Prepared data and model weights are private artifacts, not shipped in Git. The app loads:

- `data/prepared/scale-verified/test.jsonl`, `tests.json`.
- `data/external/humanevalpack.jsonl`.
- Exact merged M1 parent: `checkpoints/M1-merged-round2`.
- DPO adapter: `results/round2/remote/dpo-lr3e6/snapshots/step-66`.
- GRPO adapter: `results/round2/grpo-interim/step-800`.
- Pinned pretrained/Instruct snapshots already in the local Hugging Face cache.

Unavailable adapter choices are omitted; uncached public models fail clearly instead of silently downloading or substituting another checkpoint. Candidates execute through the existing local Bubblewrap runner with the same benchmark-specific graders. Hidden inputs, expected outputs and assertion sources are not displayed or passed to the model.

Live input/output traces stay private in `results/demo/events.jsonl`. Integration check summaries are in `results/demo/smoke-results.json` and `feedback-smoke.json`. The checks covered all5 model choices, all3 cases, editor changes, scoring, one-attempt repair and the feedback loop. The page was rendered in headless Chrome. HTTP/API operation was checked locally; access from the other computer depends on that machine's SSH/Tailscale access.
