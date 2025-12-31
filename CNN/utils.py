from model import CNN;
import torch.nn as nn;
from torch import load, device, cuda;

def load_model(path: str, device: device) -> nn.Module:
    model: nn.Module = CNN().to(device);
    state_dict: dict = load(path, weights_only=True, map_location=device);
    model.load_state_dict(state_dict);
    return (model);

def load_device() -> device:
    if (cuda.is_available()):
        device_: device = device("GPU");
    else:
        device_: device = device("cpu");
    return (device_);
