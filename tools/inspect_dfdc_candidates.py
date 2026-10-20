"""Inspect candidate file inventories/cards and metadata only; no media download."""
import csv
import io
import json
import subprocess
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from tools.audit_dfdc_mirrors import get_json
from tools.dfdc_storage import ROOT, RESEARCH, atomic_json
from tools.runtime import configure_runtime

HF=['gonnerthetooner/DFDC','gonnerthetooner/DFDC-extracted-full','laodeAlif/dfdc','dfb-data/dfdc','Shironx/DFDC_Opencv_Face_Crops','csusupergear/dfd-clips']
KAGGLE=['pranay22077/dfdc-10','khafie0212/dfdc-dataset','ashifurrahman34/dfdc-dataset','sorokin/dfdc-examples','dsabljic/dfdc-sample','aleksandrpikul222/dfdcdfdc','fakecatcherai/dfdc-dataset']


def text(url):
    try:
        with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'DFDC-research-source-audit'}),timeout=20) as response:
            raw=response.read(512*1024+1)
            return raw.decode('utf-8') if len(raw)<=512*1024 else 'Metadata/card exceeds 512 KiB'
    except OSError as error:return 'Unavailable: '+type(error).__name__


def hf(repo):
    base='https://huggingface.co/api/datasets/'+repo
    info=get_json(base+'?expand[]=cardData&expand[]=gated&expand[]=sha&expand[]=private&expand[]=downloads')
    tree=get_json(base+'/tree/main?limit=1000')
    card=text('https://huggingface.co/datasets/'+repo+'/raw/main/README.md')
    result={'repo':repo,'info':info,'root_tree':tree,'readme':card}
    atomic_json(RESEARCH/'analyses/mirror_candidates'/('hf_'+repo.replace('/','__')+'.json'),result)
    return {'repo':repo,'info':info,'root_tree':tree,'readme':card[:2500]}


def kaggle(repo):
    folder=RESEARCH/'analyses/mirror_candidates'/('kaggle_'+repo.replace('/','__'));folder.mkdir(parents=True,exist_ok=True)
    executable=str(ROOT/'dfenv/Scripts/kaggle.exe')
    listing=subprocess.run([executable,'datasets','files',repo,'--csv','--page-size','200'],capture_output=True,text=True,timeout=90)
    metadata=subprocess.run([executable,'datasets','metadata',repo,'-p',str(folder)],capture_output=True,text=True,timeout=90)
    result={'repo':repo,'list_exit':listing.returncode,'metadata_exit':metadata.returncode}
    if listing.returncode==0:
        (folder/'files_page1.csv').write_text(listing.stdout,encoding='utf-8')
        lines=listing.stdout.splitlines();start=next((i for i,line in enumerate(lines) if line.startswith('name,')),None)
        rows=list(csv.DictReader(io.StringIO('\n'.join(lines[start:])))) if start is not None else []
        result['page1_count']=len(rows);result['file_types']=dict(Counter(row['name'].split('.')[-1] for row in rows if row.get('name')));result['head']=rows[:12]
    else:result['list_error']='Request failed; no raw credentials/redirect URLs logged'
    meta=folder/'dataset-metadata.json'
    if meta.exists():
        data=json.loads(meta.read_text(encoding='utf-8'))
        result['metadata']={k:data.get(k) for k in ['title','id','licenses','description','subtitle','isPrivate','resources']}
    atomic_json(folder/'inspection.json',result);return result


def main():
    configure_runtime()
    with ThreadPoolExecutor(max_workers=3) as executor:
        h=list(executor.map(hf,HF))
        k=list(executor.map(kaggle,KAGGLE))
    github={repo:get_json('https://api.github.com/repos/'+repo+'/git/trees/HEAD?recursive=1') for repo in ['KshaunishChandalia/DFDC','itsTodFod/DFDC-Dataset']}
    result={'huggingface':h,'kaggle':k,'github':github}
    atomic_json(RESEARCH/'analyses/dfdc_candidate_inspections.json',result)
    print(json.dumps({'huggingface':h,'kaggle':k,'github':{name:{'truncated':tree.get('truncated'),'payloads':[r for r in tree.get('tree',[]) if r['path'].endswith(('.mp4','.zip','.json','.md'))][:15]} for name,tree in github.items()}},indent=2))


if __name__=='__main__':main()
