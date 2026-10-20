"""Common resident eight-frame forward benchmark; validation media, no selection."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from tools.runtime import configure_runtime
configure_runtime()
from inference.unified import ResearchPredictor,available_models
from training.video_protocol import RUN,CACHE,load_rows
from tools.dfdc_storage import atomic_json


def benchmark():
    records=[];row=load_rows('validation')[0]
    for item in available_models():
        predictor=ResearchPredictor(item['id'],'cuda')
        with np.load(CACHE/(Path(row['filename']).stem+'.npz')) as data:
            x=predictor.tensor(data[item['config']['preprocess']])
        def forward():
            if predictor.id=='S0':return predictor.legacy(x)[:,[1,0]].float().softmax(-1).mean(0)
            with torch.autocast('cuda',dtype=torch.float16):features=torch.cat([m(x) for m in predictor.encoders],-1)
            return predictor.decision(features.float()[None]).softmax(-1)
        with torch.inference_mode():
            for _ in range(5):forward()
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
            for _ in range(30):forward()
            torch.cuda.synchronize();elapsed=time.perf_counter()-tick
        paths=[Path(item['checkpoint'])]
        if predictor.id!='S0':
            paths += [Path(p) for p in predictor.spatial_saved['encoder_paths']]
            paths += [RUN/predictor.image_spatial_id/'best.pt']
            if predictor.temporal_head is not None:paths += [RUN/'S6/best.pt']
        paths=set(paths)
        records.append({'id':predictor.id,'forward_ms_per_video':elapsed/30*1000,
            'peak_vram_bytes':torch.cuda.max_memory_allocated(),'frames':8,'warmups':5,'repeats':30,
            'precision':'legacy ResNet FP32; new frozen encoders FP16 autocast and heads FP32',
            'scope':'Resident normalized GPU tensor through actual shared encoder(s), spatial/temporal heads and fusion; excludes transfer, decode, face processing, XAI',
            'media_split':'validation','filename':row['filename'],'checkpoint_sha256':predictor.checkpoint_sha256})
        records[-1]['model_artifact_bytes']=sum(p.stat().st_size for p in paths)
        records[-1]['model_artifacts']=[str(p) for p in sorted(paths)]
        print(f"benchmark {predictor.id} {records[-1]['forward_ms_per_video']:.3f} ms",flush=True)
        del predictor,x;torch.cuda.empty_cache()
    atomic_json(RUN/'runtime_benchmark.json',{'records':records,'selection_effect':'Retrospective runtime measurement only; frozen champion unchanged'})
    return records


if __name__=='__main__':benchmark()
