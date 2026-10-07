"""Create a local-path config overlay; symlink original GR00T weights unchanged."""
import argparse,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--cosmos',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    source=Path(a.checkpoint).resolve();cosmos=Path(a.cosmos).resolve();out=Path(a.out).resolve()
    if out.exists():raise FileExistsError('Choose a new overlay directory')
    # Official code dispatches its backbone using this substring.
    alias=out.parent/'nvidia/Cosmos-Reason2-2B';alias.parent.mkdir(parents=True,exist_ok=True)
    if not alias.exists():alias.symlink_to(cosmos,target_is_directory=True)
    if alias.resolve()!=cosmos:raise ValueError('Cosmos alias already points elsewhere')
    out.mkdir()
    for path in source.iterdir():
        if path.name in ('config.json','processor_config.json'):
            config=json.loads(path.read_text())
            if path.name=='config.json':config['model_name']=str(alias)
            else:config['processor_kwargs']['model_name']=str(alias)
            (out/path.name).write_text(json.dumps(config,indent=2))
        else:(out/path.name).symlink_to(path,target_is_directory=path.is_dir())
    print(out)
if __name__=='__main__':main()
