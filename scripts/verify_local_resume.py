"""Verify actual CLI resume against a completed local_content_smoke run."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('smoke_output', type=Path)
    args = parser.parse_args()
    condition = args.smoke_output.resolve() / 'label_skew_moderate'
    continuous = next((condition / 'runs/fedavg').iterdir())
    output = condition / time.strftime('resume_%H%M%S')
    output.mkdir()
    raw = yaml.safe_load((condition / 'train.yaml').read_text())
    raw['output']['root'] = str(output / 'fedavg')
    config = output / 'train.yaml'
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1',
               TEMP=str(args.smoke_output.resolve() / 'temp'), TMP=str(args.smoke_output.resolve() / 'temp'),
               RAY_TMPDIR=str(ROOT.parent / 'output/ray-local'))
    def run(name, *arguments):
        with (output / f'{name}.log').open('w', encoding='utf-8') as log:
            p = subprocess.Popen([sys.executable, '-m', 'fl_training.cli', *arguments],
                                 cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            try:
                code = p.wait(timeout=600)
            except subprocess.TimeoutExpired:
                subprocess.run(['taskkill', '/PID', str(p.pid), '/T', '/F'], capture_output=True)
                raise
        if code:
            raise RuntimeError(f'{name} exited {code}; see {output}')
        print(name + ': passed', flush=True)
    # Resume the committed round-1 snapshot of a run whose fixed budget is 2.
    # Changing the planned budget from 1 to 2 is correctly rejected by trainer.
    checkpoint = continuous / 'round_0001.pt'
    assert checkpoint.is_file()
    config.write_text(yaml.safe_dump(raw))
    run('resume', 'train', '--config', str(config), '--resume', str(checkpoint), '--no-progress')
    resumed = max((output / 'fedavg').iterdir(), key=lambda p: p.stat().st_mtime_ns)
    load = lambda p: torch.load(p, map_location='cpu', weights_only=False)
    left, right = load(continuous / 'last.pt'), load(resumed / 'last.pt')
    maximum = max(float((v.double() - right['model_state_dict'][k].double()).abs().max())
                  for k, v in left['model_state_dict'].items())
    assert maximum == 0, maximum
    run('evaluate', 'evaluate', '--checkpoint', str(resumed / 'model_final.pt'), '--output', str(output / 'evaluate'))
    result = dict(resume_max_abs_diff=maximum, tensor_entries=len(left['model_state_dict']),
                  continuous_run=str(continuous), resumed_run=str(resumed), scientific_stage2_complete=False)
    (output / 'verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
