"""Run each research stage sequentially. No dataset download or workbook creation."""
import argparse
import json
import time
from tools.runtime import configure_runtime


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true',help='Verify existing local media and prepare lossless eight-frame caches');parser.add_argument('--wait-for-cache',action='store_true');args=parser.parse_args()
    configure_runtime()
    from training.video_protocol import RUN,cache
    from training.unified_experiment import setup,extract,CONFIGS,train_one,fusion,freeze,test_all
    from tools.dfdc_storage import atomic_json,check_budget
    if args.prepare:cache()
    while args.wait_for_cache and not (RUN/'cache.json').exists():time.sleep(10)
    if not (RUN/'cache.json').exists():raise RuntimeError('Complete python -m training.video_protocol cache first')
    device=setup()
    for split in ['train','validation']:extract(split,device)
    for config in CONFIGS[1:]:train_one(config,device)
    fusion();fusion('S8','S5');freeze()
    extract('test',device);test_all(device)
    atomic_json(RUN/'training_complete.json',{'complete':True,'experiments':[c['id'] for c in CONFIGS]+['S7','S8'],'storage':check_budget()})
    print(json.dumps(json.loads((RUN/'champion.json').read_text(encoding='utf-8')),indent=2),flush=True)


if __name__=='__main__':main()
