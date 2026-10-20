"""Exercise actual Streamlit widgets, local uploads and cache behavior with AppTest."""
import csv
import json
import time
from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from config import PROJECT_ROOT
from inference.unified import ResearchPredictor
from tools.runtime import configure_runtime
from training.video_protocol import RUN


def run(output):
    configure_runtime()
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    counters = {'feature_calls':0,'attribution_calls':0}
    original_features, original_explain = ResearchPredictor.feature_sequence, ResearchPredictor.explain
    def features(self,*args,**kwargs):
        counters['feature_calls']+=1
        return original_features(self,*args,**kwargs)
    def explain(self,*args,**kwargs):
        counters['attribution_calls']+=1
        return original_explain(self,*args,**kwargs)
    timings={}; checks=[]
    with patch.object(ResearchPredictor,'feature_sequence',features), patch.object(ResearchPredictor,'explain',explain):
        app=AppTest.from_file(str(PROJECT_ROOT/'app.py'),default_timeout=60)
        def run_case(name, action=None):
            start=time.perf_counter()
            if action is not None:
                action()
            app.run()
            timings[name]=(time.perf_counter()-start)*1000
            assert not app.exception,[item.message for item in app.exception]
            footer=list(app.main.children.values())[-1]
            assert footer.label=='Notes and warnings' and not footer.proto.expanded
            assert len(app.warning)==len(footer.warning)
            checks.append(name)
            print(f'App {name}: OK ({timings[name]:.1f} ms)',flush=True)
            return footer
        def select(label):
            return next(widget for widget in app.selectbox if widget.label==label)
        run_case('initial')
        assert select('Model family').value=='ResNet / CNN'
        assert select('Trained variant').value=='S0'
        source=Path(r'C:\Users\Rangarajan\OneDrive\Pictures\Saved Pictures\random-person.jpeg')
        run_case('image_first_upload',lambda:app.file_uploader[0].set_value((source.name,source.read_bytes(),'image/jpeg')))
        assert app.metric[0].value=='FAKE'
        assert counters=={'feature_calls':1,'attribution_calls':0},counters
        first_prediction=[item.value for item in app.metric]
        run_case('image_cached_rerender')
        assert counters=={'feature_calls':1,'attribution_calls':0},counters
        run_case('gradcam_first',lambda:app.button[0].click())
        assert counters=={'feature_calls':1,'attribution_calls':1},counters
        run_case('opacity_cached',lambda:app.slider[0].set_value(.7))
        assert counters=={'feature_calls':1,'attribution_calls':1},counters
        assert [item.value for item in app.metric]==first_prediction
        run_case('target_change_no_compute',lambda:app.radio[1].set_value('REAL'))
        assert counters['attribution_calls']==1
        run_case('target_real_explanation',lambda:app.button[0].click())
        assert counters['attribution_calls']==2
        run_case('method_change_no_compute',lambda:select('Explanation method').set_value('integrated_gradients'))
        assert counters['attribution_calls']==2
        run_case('ig_on_demand',lambda:app.button[0].click())
        assert counters['attribution_calls']==3
        run_case('image_efficientnet_family',lambda:select('Model family').set_value('EfficientNet'))
        assert select('Trained variant').value=='E02_efficientnet_b0'
        assert app.metric[0].value=='FAKE'
        run_case('experimental_enabled',lambda:app.checkbox[0].set_value(True))
        run_case('experimental_temporal_family',lambda:select('Model family').set_value('Spatial + Temporal Hybrid'))
        footer=run_case('experimental_s8_fallback',lambda:select('Trained variant').set_value('S8'))
        assert any('S5 spatial checkpoint' in item.value for item in footer.info)
        run_case('video_initial',lambda:app.radio[0].set_value('Video'))
        assert select('Trained variant').value=='S6'
        with (RUN/'manifest.csv').open(newline='',encoding='utf-8') as stream:
            manifest=list(csv.DictReader(stream))
        clips=[sorted([row for row in manifest if row['split']=='validation' and row['label']==label],key=lambda row:row['filename'])[0] for label in ['REAL','FAKE']]
        for row in clips:
            source=Path(row['path'])
            run_case('video_'+row['label'],lambda:app.file_uploader[0].set_value((source.name,source.read_bytes(),'video/mp4')))
            assert app.metric[0].value==row['label']
            if row['label']=='REAL':
                before=dict(counters)
                run_case('video_explanation',lambda:app.button[0].click())
                assert counters['feature_calls']==before['feature_calls']
                assert counters['attribution_calls']==before['attribution_calls']+1
                before=dict(counters)
                run_case('video_cached_rerender')
                assert counters==before
        run_case('corrupt_video_error',lambda:app.file_uploader[0].set_value(('corrupt.mp4',b'not a video','video/mp4')))
        assert len(app.error)==1
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({'checks':checks,'milliseconds':timings,'counters':counters,
        'image_prediction':first_prediction,
        'timing_scope':'AppTest wall time includes Python UI rendering/serialization; not browser paint. First upload includes model load and kernel warmup.'},indent=2),encoding='utf-8')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=RUN.parent/'debug_refinement_20261019/app_qa.json')
    run(parser.parse_args().output)
