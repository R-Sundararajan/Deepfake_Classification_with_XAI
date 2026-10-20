"""D:-resident caches and small, dependency-free Windows resource probes."""
import ctypes
import os
import platform
import shutil
import sys
from pathlib import Path

from tools.dfdc_storage import ROOT, check_budget


def configure_runtime():
    sys.dont_write_bytecode=True
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    base = ROOT / 'data/dfdc/runtime'
    for name in ['tmp', 'torch', 'cache']:
        (base/name).mkdir(parents=True, exist_ok=True)
    os.environ['TORCH_HOME'] = str(base/'torch')
    os.environ['XDG_CACHE_HOME'] = str(base/'cache')
    os.environ['TEMP'] = os.environ['TMP'] = str(base/'tmp')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import tempfile
    tempfile.tempdir = str(base/'tmp')


def memory_snapshot():
    if sys.platform != 'win32':
        import resource
        return {'process_peak_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    from ctypes import wintypes
    class Status(ctypes.Structure):
        _fields_ = [('length',wintypes.DWORD),('load',wintypes.DWORD)] + [(n,ctypes.c_ulonglong) for n in ['total','available','page_total','page_available','virtual_total','virtual_available','extended']]
    status=Status(); status.length=ctypes.sizeof(Status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError('GlobalMemoryStatusEx failed')
    class ProcessMemory(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('faults',wintypes.DWORD)] + [(n,ctypes.c_size_t) for n in ['peak','working','pool_peak','pool','nonpool_peak','nonpool','page','page_peak']]
    counts=ProcessMemory(); counts.cb=ctypes.sizeof(counts)
    ctypes.windll.kernel32.GetCurrentProcess.restype=wintypes.HANDLE
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(ProcessMemory),wintypes.DWORD]
    ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(),ctypes.byref(counts),counts.cb)
    return {'total_ram_bytes':status.total, 'available_ram_bytes':status.available, 'process_peak_bytes':counts.peak, 'process_working_bytes':counts.working}


def audit():
    configure_runtime()
    import torch
    import torchvision
    import cv2
    report={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'torch':torch.__version__, 'torchvision':torchvision.__version__, 'opencv':cv2.__version__, 'memory':memory_snapshot(), 'storage':check_budget(), 'd_free_bytes':shutil.disk_usage(ROOT).free, 'cuda_available':torch.cuda.is_available()}
    if torch.cuda.is_available():
        free,total=torch.cuda.mem_get_info()
        report['gpu']={'name':torch.cuda.get_device_name(0),'free_bytes':free,'total_bytes':total}
    return report


if __name__=='__main__':
    import json
    from tools.dfdc_storage import atomic_json
    report=audit(); atomic_json(ROOT/'results/hardware_audit.json',report)
    print(json.dumps(report,indent=2))
