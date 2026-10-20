"""Inspect public directory/label metadata and HEAD responses; no video bytes."""
import json
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from tools.audit_dfdc_mirrors import get_json
from tools.dfdc_storage import ROOT, RESEARCH, atomic_json
from tools.runtime import configure_runtime


class HeadRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        if not newurl.startswith('https://'):raise ValueError('Non-HTTPS redirect refused')
        return urllib.request.Request(newurl,headers={'User-Agent':'DFDC-source-HEAD-audit'},method='HEAD')


def head(url):
    try:
        request=urllib.request.Request(url,headers={'User-Agent':'DFDC-source-HEAD-audit'},method='HEAD')
        with urllib.request.build_opener(HeadRedirect()).open(request,timeout=30) as response:
            return {'status':response.status,'content_type':response.headers.get('Content-Type'),'content_length':response.headers.get('Content-Length'),
                    'accept_ranges':response.headers.get('Accept-Ranges'),'etag':response.headers.get('ETag'),'final_host':urllib.parse.urlparse(response.url).hostname}
    except urllib.error.HTTPError as error:return {'status':error.code}
    except OSError as error:return {'error':type(error).__name__}


def main():
    configure_runtime()
    jobs={
        'hf_extracted_part00':'https://huggingface.co/api/datasets/gonnerthetooner/DFDC-extracted-full/tree/main/dfdc_train_part_00?limit=1000',
        'hf_archives':'https://huggingface.co/api/datasets/gonnerthetooner/DFDC/tree/main/downloads?limit=1000',
        'zenodo':'https://zenodo.org/api/records?'+urllib.parse.urlencode({'q':'"Deepfake Detection Challenge" OR "DFDC"','size':30}),
        'hf_part00_files':'https://huggingface.co/api/datasets/gonnerthetooner/DFDC-extracted-full/tree/a8a094fcdf8f10bd03971691d9b0b46445d95692/dfdc_train_part_00/dfdc_train_part_0?limit=1000',
        'hf_archive_files':'https://huggingface.co/api/datasets/gonnerthetooner/DFDC/tree/d751b7ade8ff2c06d18229d140d71644e906d678/downloads/dfdc_10?limit=1000',
        'hf_part00_metadata':'https://huggingface.co/datasets/gonnerthetooner/DFDC-extracted-full/resolve/a8a094fcdf8f10bd03971691d9b0b46445d95692/dfdc_train_part_00/dfdc_train_part_0/metadata.json',
        'hidf_official_record':'https://zenodo.org/api/records/16140829',
    }
    with ThreadPoolExecutor(max_workers=3) as executor:
        values=dict(zip(jobs,executor.map(get_json,jobs.values())))
    inspection=json.loads((RESEARCH/'analyses/dfdc_candidate_inspections.json').read_text(encoding='utf-8'))
    github=inspection['github']['itsTodFod/DFDC-Dataset']
    files=[r for r in github.get('tree',[]) if r['type']=='blob']
    values['github']={'sha':github.get('sha'),'files':len(files),'types':dict(Counter(r['path'].split('.')[-1] for r in files)),
                      'total_bytes':sum(r.get('size',0) for r in files),'nonvideo':[r for r in files if not r['path'].endswith('.mp4')]}
    if files:
        video=next(r for r in files if r['path'].endswith('.mp4'))
        url=f"https://raw.githubusercontent.com/itsTodFod/DFDC-Dataset/{github['sha']}/{video['path']}"
        values['github']['video_head']={'path':video['path'],'bytes':video['size'],**head(url)}
    for key in ['hf_extracted_part00','hf_archives']:
        rows=values[key]
        if isinstance(rows,list):
            atomic_json(RESEARCH/'analyses/mirror_candidates'/(key+'.json'),rows)
            values[key]={'entries':len(rows),'head':rows[:4],'metadata_candidates':[r for r in rows if r['path'].endswith(('.json','.txt','.md'))],
                         'directory_names':[r['path'] for r in rows if r['type']=='directory'],'total_listed_bytes':sum(r.get('size',0) for r in rows)}
    meta=values['hf_part00_metadata']
    if isinstance(meta,dict) and not meta.get('error'):
        atomic_json(RESEARCH/'analyses/mirror_candidates/hf_part00_metadata.json',meta)
        values['hf_part00_metadata']={'rows':len(meta),'labels':dict(Counter(v.get('label') for v in meta.values())),'head':list(meta.items())[:3]}
    for key in ['hf_part00_files','hf_archive_files']:
        rows=values[key]
        if isinstance(rows,list):
            atomic_json(RESEARCH/'analyses/mirror_candidates'/(key+'.json'),rows)
            values[key]={'entries':len(rows),'head':rows[:4],'metadata':[r for r in rows if r['path'].endswith('.json')],'listed_bytes':sum(r.get('size',0) for r in rows)}
    part=values['hf_part00_files']
    if isinstance(part,dict) and part.get('head'):
        video=next((r for r in part['head'] if r['path'].endswith('.mp4')),None)
        if video:
            url='https://huggingface.co/datasets/gonnerthetooner/DFDC-extracted-full/resolve/a8a094fcdf8f10bd03971691d9b0b46445d95692/'+video['path']
            values['hf_video_head']={'path':video['path'],'expected_bytes':video['size'],**head(url)}
    record=values['hidf_official_record']
    if isinstance(record,dict) and 'metadata' in record:
        values['hidf_official_record']={'title':record['metadata']['title'],'license':record['metadata'].get('license'),'access':record.get('access'),
                     'files':[{k:f.get(k) for k in ['key','size','checksum','links']} for f in record.get('files',[])]}
    z=values['zenodo']
    atomic_json(RESEARCH/'analyses/dfdc_payload_probes.json',values)
    if isinstance(z,dict):values['zenodo']={'error':z.get('error'),'total':z.get('hits',{}).get('total'),'records':[{'id':r['id'],'title':r['metadata']['title'],'access':r.get('access'),'files':[(f.get('key'),f.get('size')) for f in r.get('files',[])]} for r in z.get('hits',{}).get('hits',[])]}
    print(json.dumps(values,indent=2))


if __name__=='__main__':main()
