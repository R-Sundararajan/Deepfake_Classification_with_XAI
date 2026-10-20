"""Headless checks for both existing image and new video upload views."""
import faulthandler
from tools.runtime import configure_runtime
configure_runtime()
# Import ML extensions on the main thread before Streamlit's ScriptRunner.
from inference.media import MediaPredictor
from streamlit.testing.v1 import AppTest
from tools.dfdc_storage import ROOT, atomic_json


def main():
    faulthandler.dump_traceback_later(55,exit=True)
    print('Checking Streamlit image view',flush=True)
    app=AppTest.from_file(str(ROOT/'app.py')).run(timeout=40)
    assert not app.exception, str(app.exception)
    assert app.title[0].value=='Explainable Deepfake Classification'
    assert app.radio[0].value=='Image'
    app.radio[0].set_value('Video').run(timeout=40)
    assert not app.exception, str(app.exception)
    assert app.radio[0].value=='Video'
    app.selectbox[0].set_value('gradcam++').run(timeout=40)
    assert not app.exception, str(app.exception)
    atomic_json(ROOT/'results/smoke/app_smoke.json',{'passed':True,'views':['Image','Video'],'methods':['gradcam','gradcam++'],'scope':'Headless empty-upload view and control rerenders; actual media inference tested separately'})
    faulthandler.cancel_dump_traceback_later()
    print('Streamlit image/video views and explanation controls passed.')


if __name__=='__main__':main()
