"""Bounded concurrent XAI, with separate model instances and CUDA streams.

No Streamlit calls occur in workers. A suite reuses the original prepared input
and fixed video context; it never reclassifies or repeats face detection.
"""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import copy
from dataclasses import dataclass
import json
import threading
import time

import torch

from inference.catalog import create_predictor, methods_for


DEFAULT_PARAMETERS = {
    'integrated_gradients': {'steps': 32, 'max_steps': 128, 'baseline': 'mean'},
    'occlusion': {'patch': 32},
    'lime': {'samples': 256},
}


def clone_for_explanation(predictor):
    """Copy the loaded module graph once, preserving layer/encoder references.

    Metadata and preprocessing resources are read-only here. Workers only call
    explain_result on already prepared arrays, so no detector is duplicated.
    """
    with predictor.lock:
        clone = copy.copy(predictor)
        graph = {name: value for name, value in vars(predictor).items()
                 if isinstance(value, torch.nn.Module)
                 or (isinstance(value, list) and value and all(isinstance(item, torch.nn.Module) for item in value))}
        clone.__dict__.update(copy.deepcopy(graph))
        clone.lock = threading.RLock()
    return clone


@dataclass
class ExplanationSuite:
    futures: dict
    target: str
    started: float


class ParallelExplainer:
    def __init__(self, workers=2, max_suites=4, ttl=1800, factory=create_predictor):
        if workers < 1 or max_suites < 1:
            raise ValueError('Worker and cache limits must be positive')
        self.workers = workers
        self.max_suites = max_suites
        self.ttl = ttl
        self.factory = factory
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='xai')
        self.local = threading.local()
        self.lock = threading.RLock()
        self.suites = OrderedDict()

    def start(self, record, result, input_key, modality, positions=(), version='demo-v1', predictor=None):
        """Submit every compatible method; identical rerenders reuse futures."""
        if modality not in {'Image', 'Video'}:
            raise ValueError('Unsupported modality')
        if result['prediction'] not in {'REAL', 'FAKE'}:
            raise ValueError('Invalid prediction target')
        if modality == 'Video' and not positions:
            raise ValueError('Select at least one sampled video frame')
        model_key = (record['id'], record['checkpoint_sha256'], record['source'], record['checkpoint'], version)
        methods = tuple(methods_for(record))
        parameters_key = json.dumps(DEFAULT_PARAMETERS, sort_keys=True)
        key = (model_key, str(input_key), modality, result['prediction'], tuple(positions), methods, parameters_key)
        now = time.monotonic()
        with self.lock:
            for old_key, suite in list(self.suites.items()):
                if now - suite.started > self.ttl:
                    self._discard(old_key)
            if key in self.suites:
                self.suites.move_to_end(key)
                return self.suites[key]
            while len(self.suites) >= self.max_suites:
                self._discard(next(iter(self.suites)))
            suite = ExplanationSuite({}, result['prediction'], now)
            for method in methods:
                suite.futures[method] = self.executor.submit(
                    self._explain, model_key, record, predictor, result, modality, tuple(positions),
                    method, dict(DEFAULT_PARAMETERS.get(method, {})))
            self.suites[key] = suite
            return suite

    def _discard(self, key):
        suite = self.suites.pop(key)
        for future in suite.futures.values():
            future.cancel()

    def _explain(self, model_key, record, predictor, result, modality, positions, method, parameters):
        if getattr(self.local, 'model_key', None) != model_key:
            # One model per worker, replaced when its checkpoint changes. Never
            # share modules whose CAM/attention hooks can mutate during a call.
            self.local.predictor = None
            self.local.predictor = clone_for_explanation(predictor) if predictor is not None else self.factory(record)
            self.local.model_key = model_key
            device = self.local.predictor.device
            self.local.stream = torch.cuda.Stream(device=device) if device.type == 'cuda' else None
            if self.local.stream is not None:
                self.local.stream.wait_stream(torch.cuda.current_stream(device))
        predictor = self.local.predictor
        if modality == 'Video':
            parameters['frame_indices'] = list(positions)
        context = torch.cuda.stream(self.local.stream) if self.local.stream is not None else nullcontext()
        started = time.perf_counter()
        with predictor.lock, context:
            explanation = predictor.explain_result(result, method, result['prediction'], **parameters)
        finished = time.perf_counter()
        for item in explanation if isinstance(explanation, list) else [explanation]:
            item['details']['execution'] = {
                'worker': threading.current_thread().name,
                'started': started, 'finished': finished,
                'isolated_model': True, 'concurrent_worker_limit': self.workers,
            }
        return explanation

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
        with self.lock:
            self.suites.clear()
