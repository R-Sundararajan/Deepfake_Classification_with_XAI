"""Exercise real checkpoint inference on synthetic video; never report DFDC metrics."""
import json
import tempfile
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from evaluation.dfdc import frame_probability,load_baseline,mean_fake_probability,preprocessing_transform
from tools.acquire_dfdc import sha256
from tools.dfdc_storage import ROOT,atomic_json,check_budget
from training.dfdc_data import iter_frames


def main():
    start=time.perf_counter();torch.manual_seed(42);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable for requested GPU smoke test')
    checkpoint=ROOT/'models/best_resnet18.pth';before=sha256(checkpoint);device=torch.device('cuda')
    torch.cuda.reset_peak_memory_stats(device)
    model=load_baseline(checkpoint,device);transform=preprocessing_transform()
    base=ROOT/'results/.work';check_budget(1024*1024)
    with tempfile.TemporaryDirectory(dir=base,prefix='gpu_smoke_') as d:
        local=Path(d).resolve();assert local.is_relative_to(base.resolve())
        p=local/'synthetic.mp4';w=cv2.VideoWriter(str(p),cv2.VideoWriter_fourcc(*'mp4v'),10,(128,128))
        if not w.isOpened():raise RuntimeError('Video writer unavailable')
        rng=np.random.default_rng(42)
        for _ in range(30):w.write(rng.integers(0,256,(128,128,3),dtype=np.uint8))
        w.release()
        def evaluate():
            with torch.inference_mode():return [frame_probability(model,frame,transform,device) for _,_,frame in iter_frames(p,interval=1,max_frames=3)]
        first=evaluate();second=evaluate();torch.cuda.synchronize(device)
        assert np.allclose(first,second,rtol=0,atol=1e-7)
        assert len(first)==3 and 0<=mean_fake_probability(first)<=1
    after=sha256(checkpoint);assert before==after
    report={'test':'SYNTHETIC VIDEO GPU SMOKE — NOT DFDC','passed':True,'frames_per_pass':3,'passes':2,'deterministic_tolerance':1e-7,
            'gpu':torch.cuda.get_device_name(device),'gpu_peak_allocated_mib':torch.cuda.max_memory_allocated(device)/1024**2,
            'elapsed_seconds':time.perf_counter()-start,'checkpoint_sha256_before':before,'checkpoint_sha256_after':after,
            'torch':torch.__version__,'opencv':cv2.__version__,'haar_available':hasattr(cv2,'CascadeClassifier'),
            'note':'No ground-truth DFDC media or DFDC accuracy measured; smoke not inserted in local baseline worksheet.'}
    atomic_json(ROOT/'results/gpu_smoke_test.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
