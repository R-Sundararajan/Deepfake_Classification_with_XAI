"""Compatible methods and common presentation for the CLI and Streamlit."""
import time
import numpy as np
import torch
from explainability.gradcam import GradCAM, create_visualizations
from explainability.methods import integrated_gradients, occlusion_sensitivity, lime_explanation, attention_rollout

METHODS = {
    'gradcam': ('Grad-CAM', 'Highlights regions influencing the selected class through convolutional features.'),
    'gradcam++': ('Grad-CAM++ (gradient-power approximation)', 'Weights convolutional evidence using the common first-gradient-power approximation.'),
    'layercam': ('Layer-CAM', 'Weights each convolutional location by its positive gradient for the selected class.'),
    'integrated_gradients': ('Integrated Gradients', 'Shows signed pixel contributions to the selected class relative to a reference image.'),
    'occlusion': ('Occlusion Sensitivity', 'Measures the selected class probability change when image regions are hidden.'),
    'lime': ('LIME (local grid surrogate; slower)', 'Fits a weighted local surrogate to predictions on randomly masked image regions.'),
    'attention_rollout': ('Attention rollout', 'Visualizes transformer attention propagation; it is class agnostic and not causal proof.'),
    'gradient_attention': ('Gradient-weighted attention', 'Weights native attention by selected-class gradients; it is not causal proof.'),
}
SIGNED_METHODS = {'integrated_gradients', 'occlusion', 'lime'}


def compatible_methods(cnn=True, transformer=False):
    methods = ['gradcam', 'gradcam++', 'layercam'] if cnn else []
    return methods+['integrated_gradients', 'occlusion', 'lime']+(['attention_rollout', 'gradient_attention'] if transformer else [])


def explain(model, tensor, target, method, layer=None, transformer=None, **parameters):
    if method not in compatible_methods(layer is not None, transformer is not None):
        raise ValueError('Explanation method is incompatible with this model')
    if tensor.device.type == 'cuda':
        torch.cuda.current_stream(tensor.device).synchronize()
    started = time.perf_counter()
    if method in {'gradcam', 'gradcam++', 'layercam'}:
        generator = GradCAM(model, layer)
        try:
            cam = generator.generate(tensor, target, method)
            details = dict(generator.details)
        finally:
            generator.close()
    elif method == 'integrated_gradients':
        steps = int(parameters.get('steps', 32))
        cam, details = integrated_gradients(model, tensor, target, steps, max_steps=int(parameters.get('max_steps', max(steps, 128))), baseline=parameters.get('baseline','mean'))
    elif method == 'occlusion':
        cam, details = occlusion_sensitivity(model, tensor, target, patch=int(parameters.get('patch', 32)))
    elif method == 'lime':
        cam, details = lime_explanation(model, tensor, target, samples=int(parameters.get('samples', 256)))
    else:
        cam = attention_rollout(transformer, tensor, model, target, method == 'gradient_attention')
        details = {'class_agnostic': method == 'attention_rollout',
                   'limitation': 'Attention is not established as faithful causal attribution'}
    if tensor.device.type == 'cuda':
        torch.cuda.current_stream(tensor.device).synchronize()
    details.update(method=method, target_index=int(target), finite=bool(np.isfinite(cam).all()),
                   constant_map=bool(np.std(cam) < 1e-8), signed=method in SIGNED_METHODS,
                   attribution_ms=(time.perf_counter()-started)*1000)
    if not details['finite']:
        raise ValueError('Nonfinite attribution')
    return cam, details


def present(original, cam, method, details, strength=.45):
    started = time.perf_counter()
    signed = method in SIGNED_METHODS
    heatmap, overlay = create_visualizations(original, cam, strength, signed=signed)
    return {'original': np.asarray(original), 'heatmap': heatmap, 'overlay': overlay, 'cam': cam,
            'method': method, 'details': {**details, 'blend_ms': (time.perf_counter()-started)*1000},
            'meaning': METHODS[method][1], 'signed': signed,
            'legend': 'Blue: opposes target · white: zero · red: supports target' if signed else 'Dark: low influence · yellow: high relative influence'}
