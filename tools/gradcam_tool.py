import io
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image, UnidentifiedImageError

def gradcam_explainer(image_bytes, image_id='UNKNOWN', *, model, transform, target_layer):
    """
    Generate Grad-CAM for the frozen Model B abnormality logit.

    Returns a normalized 224x224 heatmap.
    Does not modify model weights or primary classification.
    """
    result_base = {'image_id': image_id, 'target_layer': 'backbone.layer4[-1].conv2'}
    if not isinstance(image_bytes, bytes) or not image_bytes:
        return {**result_base, 'status': 'FAILED', 'reason': 'Invalid or empty image bytes.'}
    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            image = source.convert('L')
            image.load()
    except (UnidentifiedImageError, OSError, ValueError) as error:
        return {**result_base, 'status': 'FAILED', 'reason': f'Image decoding failed: {error}'}
    activations = {}
    hook_handle = None

    def forward_hook(module, inputs, output):
        activations['value'] = output
    try:
        assert not model.training, 'Frozen Model B must be in evaluation mode.'
        assert all((not parameter.requires_grad for parameter in model.parameters())), 'Model parameters must remain frozen.'
        model_device = next(model.parameters()).device
        with torch.enable_grad():
            input_tensor = transform(image).unsqueeze(0).to(model_device).detach().requires_grad_(True)
            hook_handle = target_layer.register_forward_hook(forward_hook)
            logits = model(input_tensor)
            assert logits.numel() == 1, 'Expected one abnormality logit.'
            abnormality_logit = logits.reshape(-1)[0]
            activation_tensor = activations['value']
            gradient_tensor = torch.autograd.grad(outputs=abnormality_logit, inputs=activation_tensor, retain_graph=False, create_graph=False)[0]
            weights = gradient_tensor.mean(dim=(2, 3), keepdim=True)
            cam = (weights * activation_tensor).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode='bilinear', align_corners=False)
            cam = cam[0, 0]
            if not torch.isfinite(cam).all():
                raise ValueError('Non-finite Grad-CAM values.')
            cam_min = cam.min()
            cam_max = cam.max()
            if (cam_max - cam_min).item() > 1e-12:
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam = torch.zeros_like(cam)
            heatmap = cam.detach().cpu().numpy()
            probability = float(torch.sigmoid(abnormality_logit).detach().cpu().item())
        if heatmap.shape != (224, 224):
            raise ValueError('Unexpected Grad-CAM heatmap shape.')
        return {**result_base, 'status': 'SUCCESS', 'probability': probability, 'heatmap': heatmap, 'heatmap_shape': tuple(heatmap.shape), 'heatmap_min': float(heatmap.min()), 'heatmap_max': float(heatmap.max()), 'classification_modified': False}
    except (RuntimeError, ValueError, KeyError, TypeError, AssertionError) as error:
        return {**result_base, 'status': 'FAILED', 'reason': str(error)}
    finally:
        if hook_handle is not None:
            hook_handle.remove()
