"""Read-only public metadata discovery; never fetch video/archive payloads."""
import csv
import io
import json
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from tools.dfdc_storage import ROOT, RESEARCH, atomic_json, check_budget
from tools.runtime import configure_runtime


def get_json(url):
    request=urllib.request.Request(url,headers={'User-Agent':'DFDC-research-source-audit','Accept':'application/json'})
    try:
        with urllib.request.urlopen(request,timeout=25) as response:
            raw=response.read(8*1024**2+1)
            if len(raw)>8*1024**2:raise ValueError('Metadata response exceeds 8 MiB bound')
            return json.loads(raw)
    except urllib.error.HTTPError as error:return {'error':'HTTP '+str(error.code),'url':url}
    except (OSError,ValueError) as error:return {'error':type(error).__name__,'url':url}


def main():
    configure_runtime();check_budget(24*1024**2)
    queries={
        'huggingface': 'https://huggingface.co/api/datasets?'+urllib.parse.urlencode({'search':'dfdc','limit':100,'full':'true'}),
        'github': 'https://api.github.com/search/repositories?'+urllib.parse.urlencode({'q':'DFDC dataset','per_page':30}),
        'zenodo': 'https://zenodo.org/api/records?'+urllib.parse.urlencode({'q':'title:dfdc OR title:"Deepfake Detection Challenge"','size':30}),
    }
    result={}
    with ThreadPoolExecutor(max_workers=3) as executor:
        jobs={name:executor.submit(get_json,url) for name,url in queries.items()}
        for name,job in jobs.items():result[name]=job.result()
    result['kaggle']=[]
    executable=ROOT/'dfenv/Scripts/kaggle.exe'
    for page in [1,2,3]:
        command=[str(executable),'datasets','list','-s','dfdc','--csv','-p',str(page)]
        response=subprocess.run(command,capture_output=True,text=True,timeout=90)
        if response.returncode:
            result['kaggle_error']='Kaggle dataset-list CLI failed; no raw credential-bearing error stored';break
        lines=response.stdout.splitlines();start=next((i for i,line in enumerate(lines) if line.startswith('ref,')),None)
        if start is None:result['kaggle_error']='Unrecognized CLI CSV';break
        rows=list(csv.DictReader(io.StringIO('\n'.join(lines[start:]))))
        result['kaggle'].extend(rows)
        if len(rows)<20:break
    atomic_json(RESEARCH/'analyses/dfdc_public_source_discovery.json',result)
    hf=result['huggingface']; gh=result['github'];z=result['zenodo']
    print(json.dumps({'huggingface':[(r.get('id'),r.get('gated'),r.get('private'),r.get('tags')) for r in hf] if isinstance(hf,list) else hf,
                      'kaggle':result['kaggle'],'kaggle_error':result.get('kaggle_error'),
                      'github':[{'name':r['full_name'],'description':r.get('description')} for r in gh.get('items',[])] if isinstance(gh,dict) else gh,
                      'zenodo':[{'id':r['id'],'title':r['metadata']['title'],'files':[(f.get('key'),f.get('size')) for f in r.get('files',[])]} for r in z.get('hits',{}).get('hits',[])] if isinstance(z,dict) else z},indent=2))


if __name__=='__main__':main()
