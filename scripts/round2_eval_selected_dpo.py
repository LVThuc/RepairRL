"""Evaluate the frozen dev-selected DPO snapshot after user-stopped training."""
import json
from pathlib import Path
import subprocess
import sys
from repairrl.common import digest
from scripts.round2_train import atomic_json


def main():
    root = Path('results/round2')
    out = root / 'dpo-selected'
    out.mkdir(exist_ok=True)
    selected = json.loads((root / 'dpo-lr3e6/selection.json').read_text())['selected']
    frozen = json.loads((root / 'frozen-protocol.json').read_text())
    for path, sha in frozen['source_hashes'].items():
        if digest(path) != sha:
            raise ValueError(f'Frozen source changed: {path}')
    record = {'family': 'dpo', 'state': 'evaluating', 'selected': selected,
              'selection_set': 'fixed 60-problem dev screen',
              'note': 'User stopped further DPO optimization; use best screened checkpoint without full-dev reranking. No held-out selection.',
              'training_observation': 'Late dev degradation consistent with overfitting or objective mismatch; cause unproven.'}
    dest = out / 'selection.json'
    if dest.exists():
        old = json.loads(dest.read_text())
        if old['selected'] != selected:
            raise ValueError('Frozen DPO choice changed')
    atomic_json(dest, record)
    for controller in ('v1',):
        for benchmark in ('runbugrun', 'humanevalpack'):
            path = out / f'M2-hard-{controller}-{benchmark}.jsonl'
            if path.with_suffix('.summary.json').exists():
                continue
            subprocess.run([sys.executable, '-u', '-m', 'evaluation.round2',
                            '--model', 'checkpoints/M1-merged', '--adapter', selected['adapter'],
                            '--arm', 'M2-hard', '--controller', controller, '--benchmark', benchmark,
                            '--fp16', '--output', str(path)], check=True)
    record['state'] = 'completed'
    atomic_json(dest, record)


if __name__ == '__main__':
    main()
