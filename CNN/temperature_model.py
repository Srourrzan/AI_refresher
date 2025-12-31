import torch.nn as nn;
import torch.optim as optim;
import torch.nn.functional as F;
from torch import device, ones, no_grad, cat, Tensor;
from torch.utils.data import DataLoader;

class TemperatureScaler(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__();
        self.model: nn.Module = model;
        self.temperature: nn.Parameter = nn.Parameter(ones(1) * 1.5);
        return ;

    def forward(self, x: Tensor):
        logits = self.model(x);
        temp_logits = self.temperature_scale(logits);
        return (temp_logits);

    def temperature_scale(self, logits):
        scale = logits / self.temperature;
        return (scale);

    def set_temperature(self, valid_loader: DataLoader, device_: device, learning_rate: float = 0.01) -> float:
        logits_list: list = [];
        labels_list: list = [];
        
        if not isinstance(valid_loader, DataLoader):
            raise TypeError(f"valid_loader should be DataLoader, the current passed is {type(valid_loader).__name__}");
        if not isinstance(device_, device):
            raise TypeError(f"device should be torch.device, the current passed is {type(device).__name__}");
        self.to(device_);
        self.model.eval();
        nll_criterion = nn.CrossEntropyLoss();
        optimizer: optim.Optimizer = optim.LBFGS([self.temperature], lr=learning_rate, max_iter=50);
        with no_grad():
            for input_, label in valid_loader:
                input_ = input_.to(device_);
                logits = self.model(input_);
                logits_list.append(logits);
                labels_list.append(label.to(device_));
        logits = cat(logits_list);
        labels = cat(labels_list);

        def closure():
            optimizer.zero_grad();
            scaled: float = self.temperature_scale(logits);
            loss: Tensor = nll_criterion(scaled, labels);
            loss.backward();
            return (loss);
        
        optimizer.step(closure);
        return (self.temperature.item());
            
