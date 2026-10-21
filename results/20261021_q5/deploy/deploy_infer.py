"""CPU deployment inference for the Q7 canonical_v1 handover."""
import hashlib
from pathlib import Path

import torch
from torch import nn
from torchvision.models import mobilenet_v3_small
from torchvision import transforms
from PIL import Image, ImageOps


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(path, expected_sha256, state_key='model_state', exported=False):
    assert sha256_file(path) == expected_sha256, 'checkpoint_sha256'
    payload = torch.load(path, map_location='cpu', weights_only=exported)
    state = payload if exported else payload[state_key]
    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(1024, 38)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def preprocess(path, apply_exif=True, composite_alpha=True):
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened) if apply_exif else opened.copy()
        if composite_alpha and (image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info)):
            rgba = image.convert('RGBA')
            rgb = Image.new('RGB', rgba.size, (0, 0, 0))
            rgb.paste(rgba, mask=rgba.split()[3])
        else:
            rgb = image.convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return transform(rgb).unsqueeze(0)


def probabilities(model, path, temperature, apply_exif=True, composite_alpha=True):
    assert temperature > 0
    tensor = preprocess(path, apply_exif=apply_exif, composite_alpha=composite_alpha)
    with torch.no_grad():
        logits = model(tensor)
        return torch.softmax(logits.double() / temperature, dim=1)[0]


def predict(model, path, temperature, classes):
    probs = probabilities(model, path, temperature)
    index = int(probs.argmax())
    return {'class_id': index, 'class_name': classes[index],
            'confidence': float(probs[index]), 'confidence_percent': float(probs[index] * 100)}
