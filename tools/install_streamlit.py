"""Inspect/install the missing app dependency without changing existing packages."""
import argparse
import importlib.metadata
import json
import subprocess
import sys

from tools.dfdc_storage import ROOT, atomic_json, check_budget, tree_bytes
from tools.runtime import configure_runtime


def main(install=False):
    configure_runtime();check_budget(1024**3)
    constraints=ROOT/'results/environment_existing_constraints.txt'
    packages=sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions())
    constraints.write_text('\n'.join(f'{name}=={version}' for name,version in packages)+'\n',encoding='utf-8')
    cache=ROOT/'data/dfdc/runtime/pip_cache'
    report=ROOT/'results/streamlit_install_plan.json'
    args=[sys.executable,'-m','pip','install','streamlit==1.66.0','--only-binary=:all:','--constraint',str(constraints),
          '--cache-dir',str(cache),'--disable-pip-version-check','--report',str(report)]
    if not install:args+=['--dry-run']
    before=tree_bytes(ROOT/'dfenv')
    if install:
        atomic_json(ROOT/'results/dependency_storage.json',{'installation_complete':False,'environment_baseline_bytes':before,'environment_reserved_bytes':1024**3})
    result=subprocess.run(args)
    if result.returncode:raise SystemExit(result.returncode)
    if install:
        after=tree_bytes(ROOT/'dfenv')
        atomic_json(ROOT/'results/dependency_storage.json',{'installation_complete':True,'environment_baseline_bytes':before,'environment_after_bytes':after,
                    'environment_added_bytes':max(0,after-before),'installed':[{'name':r['metadata']['name'],'version':r['metadata']['version']} for r in json.loads(report.read_text(encoding='utf-8'))['install']]})
        for name,version in packages:
            if importlib.metadata.version(name)!=version:raise RuntimeError('An existing package version changed: '+name)
        print(f'Environment increase: {max(0,after-before)} bytes; existing package versions preserved.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--install',action='store_true')
    main(parser.parse_args().install)
