"""Standalone seminar charts generated only from recorded round-1 measurements."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DOC=Path(__file__).resolve().parents[1]/"docs/seminar"


def save(fig,name):
    fig.savefig(DOC/"assets"/(name+".svg"),bbox_inches="tight")
    fig.savefig(DOC/"assets"/(name+".png"),dpi=200,bbox_inches="tight")
    plt.close(fig)


def main():
    data=json.loads((DOC/"generated/round1-results.json").read_text())
    histories=json.loads((DOC/"generated/round1-training-curves.json").read_text())
    plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False})
    fig,axs=plt.subplots(1,2,figsize=(12,4.5),sharey=True)
    for ax,benchmark in zip(axs,["runbugrun","humanevalpack"],strict=True):
        rows=[r for r in data if r['benchmark']==benchmark]
        x=np.arange(4)
        ax.bar(x-.18,[r['metrics']['pass@1']*100 for r in rows],width=.36,label="First attempt",color="#386cb0")
        ax.bar(x+.18,[r['metrics']['agent@3']*100 for r in rows],width=.36,label="Agent@3",color="#2b9c82")
        ax.set_xticks(x,[r['arm'] for r in rows]);ax.set_title(f"{benchmark}: n={rows[0]['n']}");ax.set_ylim(0,80)
        ax.set_ylabel("All-tests-pass repair success (%)");ax.legend();ax.grid(axis='y',alpha=.2)
    fig.suptitle("Completed round 1 — fixed V1 protocol, NF4 inference, 512 generated tokens")
    save(fig,"round1-repair-success")
    panels=[("M1","loss","SFT response cross-entropy"),("M1","mean_token_accuracy","SFT teacher-forced token accuracy"),
            ("M2","loss","DPO loss"),("M2","rewards/accuracies","DPO training preference accuracy"),
            ("M3","reward","GRPO sampled training reward"),("M3","frac_reward_zero_std","GRPO groups with zero reward std")]
    fig,axs=plt.subplots(3,2,figsize=(12,10))
    for ax,(arm,key,title) in zip(axs.flat,panels,strict=True):
        rows=[r for r in histories[arm]['log_history'] if key in r]
        if rows:ax.plot([r['step'] for r in rows],[r[key] for r in rows],color="#386cb0",linewidth=1.8)
        else:ax.text(.5,.5,"Metric not recorded",ha='center',transform=ax.transAxes)
        ax.set_title(title);ax.set_xlabel("Optimizer update");ax.grid(alpha=.2)
    fig.suptitle("Logged training windows — these metrics are not held-out repair accuracy")
    fig.tight_layout(rect=[0,0,1,.97]);save(fig,"round1-training-curves")
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    arms=['M1','M2','M3']
    axs[0].bar(arms,[histories[a]['training_seconds']/60 for a in arms],color="#386cb0");axs[0].set_ylabel("Recorded run minutes")
    axs[1].bar(arms,[histories[a]['peak_allocated_gb'] for a in arms],color="#2b9c82");axs[1].set_ylabel("Peak PyTorch allocated GB")
    fig.suptitle("Round-1 RTX 3060 costs; objectives and work per update differ")
    save(fig,"round1-training-cost")


if __name__=="__main__":main()
