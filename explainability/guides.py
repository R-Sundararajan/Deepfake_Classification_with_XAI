"""Demonstration copy, separate from numerical explanation implementations."""

GUIDES = {
    'gradcam': {
        'title': 'Where convolutional features support the decision · Grad-CAM',
        'why': 'Gives a quick overview of the regions the CNN uses for this decision.',
        'depicts': 'Selected-class gradients weight the final convolutional feature maps. Bright yellow marks stronger positive influence within this map.',
    },
    'gradcam++': {
        'title': 'An alternative view of supporting regions · Grad-CAM++',
        'why': 'Checks how the localization changes under a different weighting of the same CNN features.',
        'depicts': 'A first-gradient-power approximation reweights positive CNN evidence. Yellow shows stronger support; it is a coarse feature map.',
    },
    'layercam': {
        'title': 'Location-specific convolutional support · Layer-CAM',
        'why': 'Shows how positive gradients at individual feature locations contribute to the decision.',
        'depicts': 'Each final-layer CNN location is weighted by its own positive gradient, then enlarged to the model input size. Yellow indicates stronger support.',
    },
    'integrated_gradients': {
        'title': 'Contributions relative to a reference · Integrated Gradients',
        'why': 'Separates regions that support the decision from regions that oppose it.',
        'depicts': 'Accumulates gradients along a path from a mean-color reference to the input. Red supports the explained class; blue opposes it; white is near zero.',
    },
    'occlusion': {
        'title': 'What changes when a region is hidden · Occlusion',
        'why': 'Tests whether hiding a patch changes the model score, providing a direct perturbation check.',
        'depicts': 'Hides 32-pixel patches with the mean reference color. Red means hiding that region lowered the explained-class probability; blue means it raised it.',
    },
    'lime': {
        'title': 'How masked regions relate to the score · LIME',
        'why': 'Offers a local, interpretable approximation of how regions affect this input’s prediction.',
        'depicts': 'Fits a weighted linear model to 256 randomly masked versions of a 7 × 7 grid. Red regions support the explained class in this surrogate; blue regions oppose it.',
    },
    'attention_rollout': {
        'title': 'How the transformer connects image patches · Attention rollout',
        'why': 'Shows which patches connect to the class token through the transformer layers.',
        'depicts': 'Combines native attention across layers. Yellow indicates stronger attention flow. This view is independent of the REAL/FAKE explanation target.',
    },
    'gradient_attention': {
        'title': 'Attention weighted by the decision · Gradient-weighted attention',
        'why': 'Relates transformer attention to gradients of the explained class.',
        'depicts': 'Combines attention with positive selected-class gradients. Yellow marks stronger relative gradient-weighted attention.',
    },
}


def decision_description(prediction):
    if prediction == 'REAL':
        return 'These views explain support for the model’s REAL decision. Support may be concentrated in a few regions; a real face does not require a uniform full-face heatmap.'
    return 'These views explain support for the model’s FAKE decision. A highlighted eye, mouth or other region is a place to inspect because it influenced the model.'
