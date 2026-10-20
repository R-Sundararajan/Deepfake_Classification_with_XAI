"""Fetch only the small official OpenCV Zoo YuNet detector, with provenance."""
import hashlib
import json
import urllib.request

from tools.dfdc_storage import ROOT, atomic_json, check_budget

MODEL='models/face_detection_yunet/face_detection_yunet_2023mar.onnx'


def main():
    path=ROOT/'data/dfdc/runtime/yunet.onnx';record=path.with_suffix('.json')
    if path.exists():
        meta=json.loads(record.read_text(encoding='utf-8'))
        if hashlib.sha256(path.read_bytes()).hexdigest()!=meta['sha256']: raise ValueError('Detector hash mismatch')
        print('Verified existing official YuNet detector');return
    request=urllib.request.Request('https://api.github.com/repos/opencv/opencv_zoo/commits?path='+MODEL+'&per_page=1',headers={'User-Agent':'DFDC-local-research'})
    with urllib.request.urlopen(request,timeout=30) as response: commit=json.loads(response.read(1024**2))[0]['sha']
    url=f'https://media.githubusercontent.com/media/opencv/opencv_zoo/{commit}/{MODEL}'
    check_budget(1024**2)
    with urllib.request.urlopen(url,timeout=60) as response: data=response.read(1024**2+1)
    if not 100_000<len(data)<1024**2: raise ValueError('Unexpected YuNet model payload/size')
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.part')
    with temporary.open('xb') as stream: stream.write(data)
    temporary.replace(path)
    atomic_json(record,{'source':url,'commit':commit,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),
                        'documentation':'https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet'})
    print(f'Saved official YuNet detector: {len(data)} bytes')


if __name__=='__main__': main()
