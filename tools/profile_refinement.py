"""Synchronized before/after resident-model measurements and baseline sensitivity."""
import ast
import importlib.util
import json
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import streamlit as st
from config import PROJECT_ROOT
from inference.unified import ResearchPredictor
from inference.catalog import catalog, create_predictor
from inference.predict import DeepfakePredictor
from tools.diagnose_refinement import timed
from tools.verify_refinement import preprocessing_checks
from tools.runtime import configure_runtime

OUTPUT=PROJECT_ROOT/'results/debug_refinement_20261019'
PICTURES=Path(r'C:\Users\Rangarajan\OneDrive\Pictures\Saved Pictures')


def load_source(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def samples(call,repeats=7):
    for _ in range(2):call()
    results=[timed(call)[1] for _ in range(repeats)]
    return {'median_ms':float(np.median(results)),'min_ms':min(results),'max_ms':max(results),'repeats':repeats}


def run(output=OUTPUT):
    configure_runtime();torch.set_num_threads(2)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    destination=output/'profile.json'
    if destination.exists():raise FileExistsError(destination)
    snapshot=OUTPUT/'baseline/source_snapshot'
    before=load_source('before_unified',snapshot/'inference/unified.py')
    old_cam=load_source('before_cam',snapshot/'explainability/gradcam.py')
    old_methods=load_source('before_methods',snapshot/'explainability/methods.py')
    before.GradCAM=old_cam.GradCAM;before.create_visualizations=old_cam.create_visualizations
    before.integrated_gradients=old_methods.integrated_gradients
    before.attention_rollout=old_methods.attention_rollout
    report={'scope':'Same process/device, synchronized GPU, resident models, two warmups and seven repeats. Browser paint excluded.',
            'measurements':[],'baseline_comparison':[]}
    records={row['id']:row for row in catalog('Image')+catalog('Video')}
    for identifier in ['S0','S2']:
        old,load_before=timed(lambda:before.ResearchPredictor(identifier))
        new,load_after=timed(lambda:ResearchPredictor(identifier))
        for name in ['random-person.jpeg','real.jpeg']:
            image=Image.open(PICTURES/name).convert('RGB')
            original=samples(lambda:old.predict_image(image,'gradcam'))
            corrected=samples(lambda:new.predict_image(image,'gradcam'))
            prediction=samples(lambda:new.predict_image(image,None))
            result=new.predict_image(image,None)
            report['measurements'].append({'model':identifier,'input':name,'baseline_gradcam':original,
                'corrected_gradcam':corrected,'prediction_only':prediction,'stages_ms':result['timings'],
                'checkpoint_load_before_ms':load_before,'checkpoint_load_after_ms':load_after})
            if identifier=='S2':
                for baseline in ['mean','blur']:
                    explanation,elapsed=timed(lambda:new.explain_result(result,'integrated_gradients',baseline=baseline,max_steps=256))
                    report['baseline_comparison'].append({'input':name,'baseline':baseline,'milliseconds':elapsed,'details':explanation['details']})
                    Image.fromarray(explanation['overlay']).save(output/f'IG_S2_{Path(name).stem}_{baseline}_256.png')
        del old,new
        torch.cuda.empty_cache()
    # Compile just the actual app's decorated cache functions, without executing
    # its page. Bare-mode memory caches use the same Streamlit decorators.
    source=ast.parse((PROJECT_ROOT/'app.py').read_text(encoding='utf-8'))
    functions=[node for node in source.body if isinstance(node,ast.FunctionDef) and node.name in {'load_predictor','classify','cached_explanation'}]
    environment={'st':st,'Image':Image,'create_predictor':create_predictor,'json':json}
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(PROJECT_ROOT/'app.py'),'exec'),environment)
    cached=[]
    for identifier in ['S0','S2']:
        record=records[identifier]
        args=(identifier,record['checkpoint_sha256'],'unified',record['checkpoint'],str(PICTURES/'random-person.jpeg'),'Image','profile-v1')
        result=environment['classify'](*args)
        prediction_cache=samples(lambda:environment['classify'](*args))
        xai_args=args[:-1]+('FAKE','gradcam','{}',(),args[-1])
        environment['cached_explanation'](*xai_args)
        explanation_cache=samples(lambda:environment['cached_explanation'](*xai_args))
        cached.append({'model':identifier,'prediction_cache':prediction_cache,'explanation_cache':explanation_cache})
    report['streamlit_memory_cache']=cached
    preprocessed=preprocessing_checks(output/'preprocessing')
    s0=ResearchPredictor('S0')
    cli=DeepfakePredictor()
    for name in ['random-person.jpeg','real.jpeg']:
        image=Image.open(PICTURES/name).convert('RGB')
        old=np.asarray(image.resize((224,224)))
        p,_,_=s0.score(s0.feature_sequence(old[None]),image=True)
        fixed=cli.predict(image)
        preprocessed.append({'model':'S0','input':name,'preprocessing':'legacy CLI before: implicit Pillow bicubic resize','fake_probability':p})
        preprocessed.append({'model':'S0','input':name,'preprocessing':'CLI corrected: explicit bilinear, matching training/Streamlit','fake_probability':fixed['fake_probability']})
    (output/'preprocessing/checks.json').write_text(json.dumps(preprocessed,indent=2),encoding='utf-8')
    destination.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'measurements':report['measurements'],'cache':cached},indent=2),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    run(parser.parse_args().output)
