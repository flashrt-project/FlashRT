"""Attach a Qwen tokenizer and embodiment normalization to XR-1 5B weights."""
import argparse
import json
from pathlib import Path
import shutil

import yaml


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True, help='Directory containing model_states.pt')
    p.add_argument('--tokenizer', required=True, help='Local Qwen3-VL-4B-Instruct processor snapshot')
    p.add_argument('--stats', required=True, help='JSON statistics or upstream data YAML')
    p.add_argument('--embodiment', required=True, help='Name for this robot action schema')
    a = p.parse_args()
    root = Path(a.checkpoint)
    if not (root / 'model_states.pt').is_file():
        p.error('checkpoint lacks model_states.pt')
    source = Path(a.tokenizer)
    if not (source / 'config.json').is_file():
        p.error('tokenizer snapshot lacks config.json')
    stats_path = Path(a.stats)
    if stats_path.suffix.lower() in ('.yaml', '.yml'):
        stats = yaml.safe_load(stats_path.read_text())['data']['params']['train_datasets']
    else:
        stats = json.loads(stats_path.read_text())
    stats = {k: stats[k] for k in ('mean', 'std', 'q01', 'q99')}
    stats['embodiment'] = a.embodiment
    target = root / 'tokenizer'
    target.mkdir(exist_ok=True)
    for src in source.iterdir():
        if src.is_file() and src.suffix in ('.json', '.txt', '.jinja'):
            shutil.copy2(src, target / src.name)
    (root / 'normalization.json').write_text(json.dumps(stats, indent=2) + '\n')
    from flash_rt.amd.models.xr1.checkpoint_rdna35 import XR1FoundationProcessor
    processor = XR1FoundationProcessor(root)
    print('Prepared XR-1 5B:', root, processor.list_robot_types())


if __name__ == '__main__':
    main()
