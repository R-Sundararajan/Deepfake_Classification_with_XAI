"""Input attribution and native attention; signed maps retain opposing evidence."""
import numpy as np
import torch
from torch.nn import functional as F


def normalize_map(value):
    value = np.asarray(value, dtype=np.float32)
    if not np.isfinite(value).all():
        raise ValueError('Nonfinite attribution')
    maximum = float(np.max(np.abs(value)))
    return value / maximum if maximum > 1e-12 else np.zeros_like(value)


def statistics(value):
    x = np.asarray(value, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite attribution')
    return {'min': float(x.min()), 'max': float(x.max()), 'mean': float(x.mean()), 'std': float(x.std()),
            'abs_sum': float(np.abs(x).sum()), 'signed_sum': float(x.sum()),
            'positive_sum': float(x[x > 0].sum()), 'negative_sum': float(x[x < 0].sum()),
            'absolute_quantiles_0_50_90_99_100': np.quantile(np.abs(x), [0, .5, .9, .99, 1]).tolist()}


def integrated_gradients(model, tensor, class_index, steps=32, max_steps=None, batch_size=8, baseline='mean'):
    if tensor.shape[0] != 1 or steps < 2 or batch_size < 1:
        raise ValueError('IG requires one image, at least two steps and a positive batch size')
    max_steps = max_steps or steps
    if max_steps < steps:
        raise ValueError('IG maximum steps must cover initial steps')
    if baseline == 'mean':
        baseline = torch.zeros_like(tensor)
        baseline_name = 'zero normalized input = ImageNet mean RGB'
    elif baseline == 'blur':
        kernel = min(31, min(tensor.shape[-2:])//2*2-1)
        padding = kernel//2
        baseline = F.avg_pool2d(F.pad(tensor,(padding,padding,padding,padding),mode='reflect'),kernel,1).detach()
        baseline_name = f'input blurred with a {kernel}-pixel mean filter; explains differences from this reference'
    else:
        raise ValueError('Unknown IG reference baseline')
    with torch.no_grad():
        difference = float(model(tensor)[0, class_index] - model(baseline)[0, class_index])
    history = []
    while True:
        nodes, weights = np.polynomial.legendre.leggauss(steps)
        total = torch.zeros_like(tensor)
        gradient_abs_sum = 0.
        with torch.enable_grad():
            for start in range(0, steps, batch_size):
                alpha = tensor.new_tensor((nodes[start:start+batch_size]+1)/2).reshape(-1, 1, 1, 1)
                weight = tensor.new_tensor(weights[start:start+batch_size]/2).reshape(-1, 1, 1, 1)
                point = (baseline+(tensor-baseline)*alpha).detach().requires_grad_(True)
                gradient = torch.autograd.grad(model(point)[:, class_index].sum(), point)[0]
                if not torch.isfinite(gradient).all():
                    raise ValueError('Nonfinite IG gradient')
                total += (gradient*weight).sum(0, keepdim=True)
                gradient_abs_sum += float(gradient.detach().abs().sum())
        attribution = ((tensor-baseline)*total).detach()
        residual = abs(difference-float(attribution.sum()))
        relative = residual/max(abs(difference), 1e-8)
        history.append({'steps': steps, 'absolute_residual': residual, 'relative_residual': relative})
        converged = residual <= .01+.05*abs(difference)
        if converged or steps >= max_steps:
            break
        steps = min(steps*2, max_steps)
    raw = attribution.cpu().numpy()[0]
    signed = raw.sum(0)
    detail = {'steps': steps, 'quadrature': 'Gauss-Legendre', 'internal_batch_size': batch_size,
              'baseline': baseline_name, 'output_difference': difference,
              'signed_attribution_sum': float(attribution.sum()), 'completeness_absolute_residual': residual,
              'completeness_relative_residual': relative, 'converged': converged,
              'convergence_tolerance': 'absolute <= 0.01 + 0.05 * abs(output difference)',
              'convergence_history': history, 'raw_attribution_statistics': statistics(raw),
              'gradient_abs_sum': gradient_abs_sum, 'gradient_available': gradient_abs_sum > 0,
              'low_signal': float(np.abs(raw).sum()) <= 1e-8, 'target_index': int(class_index),
              'display': 'signed channel sum; red supports and blue opposes the selected class relative to the baseline'}
    return normalize_map(signed), detail


def occlusion_sensitivity(model, tensor, class_index, patch=32, batch_size=8):
    if tensor.shape[0] != 1 or not 8 <= patch <= min(tensor.shape[-2:]):
        raise ValueError('Occlusion requires one image and a valid patch size')
    height, width = tensor.shape[-2:]
    regions = [(y, min(y+patch, height), x, min(x+patch, width))
               for y in range(0, height, patch) for x in range(0, width, patch)]
    raw = np.zeros((height, width), np.float32)
    with torch.inference_mode():
        clean = float(model(tensor).softmax(-1)[0, class_index])
        for start in range(0, len(regions), batch_size):
            batch = tensor.repeat(len(regions[start:start+batch_size]), 1, 1, 1)
            for item, (y1, y2, x1, x2) in enumerate(regions[start:start+batch_size]):
                batch[item, :, y1:y2, x1:x2] = 0
            drops = clean-model(batch).float().softmax(-1)[:, class_index].cpu().numpy()
            for drop, (y1, y2, x1, x2) in zip(drops, regions[start:start+batch_size]):
                raw[y1:y2, x1:x2] = drop
    return normalize_map(raw), {'patch_size': patch, 'regions': len(regions), 'baseline': 'ImageNet mean RGB',
                                'raw_attribution_statistics': statistics(raw), 'clean_class_probability': clean,
                                'limitation': 'Masking may create out-of-distribution inputs', 'signed': True}


def lime_explanation(model, tensor, class_index, samples=256, grid=7, batch_size=16, seed=42):
    from lime.lime_base import LimeBase
    from sklearn.metrics import pairwise_distances
    if tensor.shape[0] != 1 or samples < grid*grid+2:
        raise ValueError('LIME needs one image and more samples than grid regions')
    rng = np.random.RandomState(seed)
    count = grid*grid
    masks = rng.randint(0, 2, (samples, count))
    masks[0] = 1
    height, width = tensor.shape[-2:]
    yy = np.minimum(np.arange(height)*grid//height, grid-1)
    xx = np.minimum(np.arange(width)*grid//width, grid-1)
    segments = yy[:, None]*grid+xx[None, :]
    predictions = []
    with torch.inference_mode():
        for start in range(0, samples, batch_size):
            keep = tensor.new_tensor(masks[start:start+batch_size, segments])[:, None]
            predictions.append(model(tensor*keep).float().softmax(-1).cpu().numpy())
    predictions = np.concatenate(predictions)
    distances = pairwise_distances(masks, masks[0:1], metric='cosine').ravel()*100
    kernel = lambda distance: np.sqrt(np.exp(-(distance**2)/(25.**2)))
    surrogate = LimeBase(kernel, random_state=seed)
    intercept, coefficients, fit, local = surrogate.explain_instance_with_data(
        masks, predictions, distances, class_index, count, feature_selection='none')
    weights = np.zeros(count, np.float32)
    for index, coefficient in coefficients:
        weights[index] = coefficient
    raw = weights[segments]
    return normalize_map(raw), {'samples': samples, 'seed': seed, 'regions': count,
        'segmentation': f'{grid} by {grid} nonoverlapping grid; not semantic objects',
        'weighted_surrogate_r2': float(fit), 'surrogate_intercept': float(intercept),
        'surrogate_local_prediction': float(np.asarray(local).ravel()[0]),
        'actual_target_probability': float(predictions[0, class_index]),
        'raw_attribution_statistics': statistics(weights), 'signed': True,
        'limitation': 'Local surrogate approximation; poor fit and unrealistic masks limit interpretation'}


def attention_rollout(model, tensor, decision_model=None, class_index=None, gradient_weighted=False):
    matrices, handles, previous = [], [], []
    try:
        for block in model.blocks:
            previous.append(block.attn.fused_attn)
            block.attn.fused_attn = False
            handles.append(block.attn.attn_drop.register_forward_hook(lambda module, inputs, output: matrices.append(output)))
        if gradient_weighted:
            if decision_model is None or class_index is None:
                raise ValueError('Gradient attention requires the selected decision and target')
            with torch.enable_grad():
                x = tensor.detach().clone().requires_grad_(True)
                score = decision_model(x)[:, class_index].sum()
                gradients = torch.autograd.grad(score, matrices)
            matrices = [(attention.detach()*gradient.detach()).clamp_min(0) for attention, gradient in zip(matrices, gradients)]
        else:
            with torch.inference_mode():
                model(tensor)
            matrices = [matrix.detach() for matrix in matrices]
        if not matrices:
            raise RuntimeError('No native attention tensors captured')
        identity = torch.eye(matrices[0].shape[-1], device=tensor.device)
        joint = identity
        for matrix in matrices:
            attention = matrix[0].float().mean(0)+identity
            attention = attention/attention.sum(-1, keepdim=True)
            joint = attention@joint
        tokens = joint[0, 1:]
        side = int(tokens.numel()**.5)
        if side*side != tokens.numel():
            raise ValueError('Unexpected token grid')
        result = F.interpolate(tokens.reshape(1, 1, side, side), tensor.shape[-2:], mode='bilinear', align_corners=False)[0, 0]
        return normalize_map(result.detach().cpu().numpy())
    finally:
        for handle in handles:
            handle.remove()
        for block, value in zip(model.blocks, previous):
            block.attn.fused_attn = value


def check_attribution(model, tensor, target, cam, fraction=.15, seed=42):
    """Bounded deletion versus random masking; keep negative results visible."""
    flat = np.asarray(cam).ravel()
    k = max(1, int(len(flat)*fraction))
    rng = np.random.default_rng(seed)
    def score(indices=None):
        x = tensor.clone()
        if indices is not None:
            mask = torch.zeros(len(flat), dtype=torch.bool, device=x.device)
            mask[indices.copy()] = True
            x = x.masked_fill(mask.reshape(1, 1, *cam.shape), 0.)
        with torch.inference_mode():
            return float(model(x).float().softmax(-1)[0, target])
    clean = score()
    top = score(np.argsort(flat, kind='stable')[-k:])
    random_drop = np.mean([clean-score(rng.choice(len(flat), k, False)) for _ in range(3)])
    return {'clean_class_probability': clean, 'mask_fraction': fraction, 'salient_deletion_drop': clean-top,
            'random_deletion_drop': float(random_drop), 'advantage': float(clean-top-random_drop),
            'nonconstant': bool(np.std(cam) > 1e-8), 'finite': bool(np.isfinite(cam).all()),
            'limitation': 'Mean-color masks may be out of distribution; a small sanity check is not proof of faithfulness'}
