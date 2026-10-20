"""Sequential CUDA feasibility checks and legacy media inference; no DFDC metrics."""
import gc
import json
import time
from pathlib import Path

from tools.runtime import configure_runtime
configure_runtime()
import cv2
import numpy as np
import torch
from PIL import Image

from explainability.evaluate_cam import evaluate_map
from inference.media import MediaPredictor, save_media_result
from models.transfer import TransferClassifier
from tools.acquire_dfdc import sha256
from tools.dfdc_storage import ROOT, atomic_json, check_budget
from tools.runtime import memory_snapshot
from training.dfdc_cache import YuNetCropper


def main():
    torch.set_num_threads(2);torch.manual_seed(42)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output=ROOT/'results/smoke';output.mkdir(parents=True,exist_ok=True)
    results=[]
    cases=[('resnet18','none','none',0),('efficientnet_b0','none','none',0),('mobilenet_v3_large','none','none',0),
           ('resnet18','eca','none',0),('resnet18','cbam','none',0),('resnet18','none','frequency',0),('resnet18','none','none',1)]
    for backbone,attention,fusion,late_blocks in cases:
        gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter();model=TransferClassifier(backbone,attention,fusion,pretrained=False).to(device);model.set_trainable(late_blocks);model.train()
        inputs=torch.randn(2,3,224,224,device=device);targets=torch.tensor([0,1],device=device)
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
        with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
            logits=model(inputs);loss=torch.nn.functional.cross_entropy(logits,targets)
        loss.backward();optimizer.step()
        assert tuple(logits.shape)==(2,2) and torch.isfinite(loss)
        if device.type=='cuda':torch.cuda.synchronize()
        row={'backbone':backbone,'added_attention':attention,'fusion':fusion,'fine_tune_blocks':late_blocks,'batch_size':2,'input_size':224,
             'seconds':time.perf_counter()-start,'peak_gpu_mib':torch.cuda.max_memory_allocated()/1024**2 if device.type=='cuda' else None,
             'parameters':sum(p.numel() for p in model.parameters()),'memory':memory_snapshot(),'passed':True}
        results.append(row);print(json.dumps({k:row[k] for k in ['backbone','added_attention','fusion','fine_tune_blocks','peak_gpu_mib','passed']}),flush=True)
        del model,optimizer,inputs,targets,logits,loss
    gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    checkpoint=ROOT/'models/best_resnet18.pth';before=sha256(checkpoint)
    predictor=MediaPredictor(checkpoint,device)
    video=output/'synthetic_media.mp4'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),4,(128,128))
    if not writer.isOpened():raise RuntimeError('Synthetic video encoder missing')
    for i in range(12):writer.write(np.random.default_rng(i).integers(0,255,(128,128,3),dtype=np.uint8))
    writer.release()
    for method in ['gradcam','gradcam++']:
        result=predictor.predict_video(video,method,explanations=1)
        assert len(result['sampled_frames'])==3 and len(result['explanations'])==1
        assert np.isfinite(result['explanations'][0]['cam']).all()
    image=Image.fromarray(np.random.default_rng(42).integers(0,255,(128,128,3),dtype=np.uint8))
    prediction=predictor.predict_image(image)
    if not (output/'media_demo').exists():save_media_result(prediction,output/'media_demo')
    tensor=predictor.transform(image).unsqueeze(0).to(device)
    xai=evaluate_map(predictor.model,predictor.layer,tensor,0)
    cropper=YuNetCropper(ROOT/'data/dfdc/runtime/yunet.onnx')
    crop,mode=cropper(np.zeros((240,320,3),np.uint8));assert mode=='no_face_full_frame_fallback'
    # Read one already-existing face image only to verify landmark/detector code.
    sample=next((ROOT/'data/real_vs_fake/real-vs-fake/test').rglob('*.jpg'),None)
    face_status='No existing JPEG available'
    if sample:
        bgr=cv2.imread(str(sample));crop,face_status=cropper(bgr)
        assert crop.size and np.isfinite(crop).all()
    assert sha256(checkpoint)==before
    report={'passed':True,'dataset':'synthetic software fixtures; no DFDC evaluation','pretrained':False,
            'initialization':'random weights solely for forward/backward memory tests; legacy inference uses unchanged existing checkpoint',
            'device':str(device),'architectures':results,'legacy_checkpoint_sha256':before,'image_and_video_inference_passed':True,
            'cam_methods':['gradcam','gradcam++'],'yunet_existing_face_status':face_status,'synthetic_xai_function_check':xai,'storage':check_budget()}
    atomic_json(output/'architecture_smoke.json',report)
    print('All sequential architecture, media and attribution smoke checks passed.',flush=True)


if __name__=='__main__':main()
