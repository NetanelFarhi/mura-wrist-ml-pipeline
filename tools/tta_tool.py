"""Reusable TTA tools for MURA Agentic AI.

Uses frozen ResNet18 Model B.
Requires explicit model, preprocessing, and threshold dependencies.
Rejects non-canonical classification thresholds.
Does not modify primary study-level classification.
"""

import io
import numpy as np
import torch

from PIL import Image, ImageEnhance, UnidentifiedImageError

def tta_uncertainty(image_bytes, image_id='UNKNOWN', *, resnet18_model, inference_transform, selected_threshold):
    """
    Evaluate frozen ResNet18 prediction stability under
    five deterministic image variants.

    Supplementary analysis only; does not modify
    the classifier, threshold, or study-level prediction.
    """
    if isinstance(selected_threshold, (bool, np.bool_)) or not np.isfinite(float(selected_threshold)) or (not np.isclose(float(selected_threshold), 0.263567715883255, rtol=0, atol=1e-12)):
        raise ValueError('Frozen Model B threshold mismatch.')
    if not isinstance(image_bytes, bytes) or not image_bytes:
        return {'image_id': image_id, 'status': 'FAILED', 'reason': 'Invalid or empty image bytes'}
    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            image = source.convert('L')
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return {'image_id': image_id, 'status': 'BLOCKED', 'reason': f'Image decoding failed: {exc}'}
    augmented_images = {'original': image, 'brightness_plus_5': ImageEnhance.Brightness(image).enhance(1.05), 'brightness_minus_5': ImageEnhance.Brightness(image).enhance(0.95), 'contrast_plus_5': ImageEnhance.Contrast(image).enhance(1.05), 'contrast_minus_5': ImageEnhance.Contrast(image).enhance(0.95)}
    model_device = next(resnet18_model.parameters()).device
    resnet18_model.eval()
    predictions = {}
    with torch.inference_mode():
        for name, augmented_image in augmented_images.items():
            tensor = inference_transform(augmented_image).unsqueeze(0).to(model_device)
            logits = resnet18_model(tensor)
            probability = torch.sigmoid(logits).reshape(-1).item()
            predictions[name] = float(probability)
    probabilities = np.asarray(list(predictions.values()), dtype=np.float64)
    assert len(probabilities) == 5
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    threshold_classes = probabilities >= selected_threshold
    threshold_disagreement = bool(np.any(threshold_classes) and np.any(~threshold_classes))
    return {'image_id': image_id, 'status': 'SUCCESS', 'original_probability': predictions['original'], 'tta_probabilities': predictions, 'tta_mean_probability': float(np.mean(probabilities)), 'tta_std': float(np.std(probabilities, ddof=0)), 'tta_min_probability': float(np.min(probabilities)), 'tta_max_probability': float(np.max(probabilities)), 'tta_range': float(np.ptp(probabilities)), 'threshold_disagreement': threshold_disagreement, 'threshold': float(selected_threshold), 'augmentation_count': len(predictions)}

def interpret_tta_result(tta_result, *, selected_threshold):
    """
    Convert TTA probabilities into descriptive decision support.
    Does not run inference or modify classification.
    """
    if isinstance(selected_threshold, (bool, np.bool_)) or not np.isfinite(float(selected_threshold)) or (not np.isclose(float(selected_threshold), 0.263567715883255, rtol=0, atol=1e-12)):
        raise ValueError('Frozen Model B threshold mismatch.')

    def unavailable(reason):
        return {'status': 'UNAVAILABLE', 'stability_category': 'UNAVAILABLE', 'reason': reason}
    if not isinstance(tta_result, dict):
        return unavailable('TTA result is not a dictionary.')
    if tta_result.get('status') != 'SUCCESS':
        return unavailable('TTA execution was unsuccessful.')
    try:
        predictions = tta_result['tta_probabilities']
        required_augmentations = {'original', 'brightness_plus_5', 'brightness_minus_5', 'contrast_plus_5', 'contrast_minus_5'}
        if not isinstance(predictions, dict) or set(predictions) != required_augmentations:
            raise ValueError('Missing or unexpected augmentations.')

        def valid_probability(value):
            if isinstance(value, (bool, np.bool_)):
                raise ValueError('Boolean probability is invalid.')
            number = float(value)
            if not np.isfinite(number) or not 0 <= number <= 1:
                raise ValueError('Invalid probability value.')
            return number
        probabilities = np.asarray([valid_probability(v) for v in predictions.values()], dtype=np.float64)
        threshold = float(tta_result['threshold'])
        if not np.isfinite(threshold) or not np.isclose(threshold, selected_threshold, rtol=0, atol=1e-12):
            raise ValueError('Frozen threshold mismatch.')
        original_probability = valid_probability(tta_result['original_probability'])
        if not np.isclose(original_probability, valid_probability(predictions['original']), rtol=0, atol=1e-08):
            raise ValueError('Original probability mismatch.')
        probability_range = float(np.ptp(probabilities))
        probability_std = float(np.std(probabilities, ddof=0))
        threshold_crossing = bool(np.any(probabilities >= threshold) and np.any(probabilities < threshold))
        if tta_result.get('augmentation_count') != 5:
            raise ValueError('Invalid augmentation count.')
        for field, expected in [('tta_range', probability_range), ('tta_std', probability_std)]:
            reported = float(tta_result[field])
            if not np.isfinite(reported) or not np.isclose(reported, expected, rtol=0, atol=1e-08):
                raise ValueError(f'Inconsistent {field}.')
        reported_crossing = tta_result['threshold_disagreement']
        if not isinstance(reported_crossing, (bool, np.bool_)) or bool(reported_crossing) != threshold_crossing:
            raise ValueError('Inconsistent threshold disagreement.')
        category = 'THRESHOLD_CROSSING' if threshold_crossing else 'NO_THRESHOLD_CROSSING'
        explanation = 'Augmented predictions crossed the frozen classification threshold.' if threshold_crossing else 'All augmented predictions remained on the same side of the frozen classification threshold.'
        return {'status': 'SUCCESS', 'stability_category': category, 'original_probability': original_probability, 'tta_probability_range': probability_range, 'tta_probability_std': probability_std, 'threshold': threshold, 'threshold_crossing': threshold_crossing, 'explanation': explanation, 'classification_modified': False}
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        return unavailable(str(error))
