"""Final cache/source integrity and face-fallback accounting, without predictions."""
import json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from tools.acquire_dfdc import sha256
from tools.dfdc_storage import atomic_json,check_budget
from training.video_protocol import RUN,CACHE,load_rows,normalized_indices
from training.dfdc_data import load_metadata,source_groups


def audit(verify_sources=True):
    rows=load_rows();methods=defaultdict(Counter);sources={};full_metadata=load_metadata('data/dfdc/metadata/metadata.json')
    byname={r['filename']:r for r in rows}
    for group,names in source_groups(full_metadata).items():
        if len({byname[n]['split'] for n in names if n in byname})>1:raise ValueError('Full metadata graph exposes split leakage')
    for row in rows:
        if verify_sources and sha256(row['path'])!=row['sha256']:raise ValueError('Original media changed since manifest')
        with np.load(CACHE/(Path(row['filename']).stem+'.npz'),allow_pickle=False) as data:
            if str(data['source_sha256'])!=row['sha256']:raise ValueError('Cache source mismatch')
            if data['raw'].shape!=(8,224,224,3) or data['face'].shape!=(8,224,224,3):raise ValueError('Cache shape mismatch')
            if data['indices'].tolist()!=normalized_indices(int(row['frame_count'])):raise ValueError('Cache sampled positions differ')
            methods[row['split']+'_'+row['label']].update(data['methods'].tolist())
        sources.setdefault(row['source_group'],set()).add(row['split'])
    if any(len(s)>1 for s in sources.values()):raise ValueError('Source leakage')
    record={'passed':True,'videos':len(rows),'source_hashes_reverified':verify_sources,
            'full_metadata_graph_disjoint':True,'face_methods_by_split_class':{k:dict(v) for k,v in methods.items()},'storage':check_budget()}
    atomic_json(RUN/'cache_integrity.json',record);print(json.dumps(record,indent=2),flush=True)
    return record


if __name__=='__main__':audit()
