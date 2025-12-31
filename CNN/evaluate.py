import torch;
import numpy as np;
import matplotlib.pyplot as plt;
import torch.nn.functional as functional;
from sklearn.metrics import brier_score_loss;
from sklearn.calibration import calibration_curve, CalibratedClassifierCV;

from model import CNN;
from utils import load_model, load_device;
from data_loader import get_val_dataloader, DataLoader;
from temperature_model import TemperatureScaler;

def multiclass_ece(probs, labels, n_bins=15) -> float:
    ece: float = 0.0;
    n: int = len(labels);
    bins: np.ndarray = np.linspace(0, 1, n_bins + 1);
    print(f"probs.shape = {probs.shape}");
    # for c in range(probs.shape[0]):
    #     conf: list = probs[:, c];
    #     true_mask: bool = (labels == c);
    #     for i in range(n_bins):
    #         mask: bool = (conf >= bins[i]) & (conf < bins[i + 1]);
    #         if mask.sum() > 0:
    #             acc: float = true_mask[mask].mean();
    #             avg_conf: float = conf[mask].mean();
    #             ece += (mask.sum() / n) * abs(acc - avg_conf);
    return (ece);


def plot_calibration(fop, mpv):
    plt.plot(mpv, fop, marker='o');
    plt.plot([0, 1], [0, 1], linestyle='--');
    plt.title("Calibration Curve");
    plt.xlabel("Mean Predicted Value");
    plt.ylabel("Fraction of Positives");
    plt.show();

    
def calibration_metrics(model: torch.nn.Module, val_loader: DataLoader, device: torch.device):
    """
    To make the output of the model represent probabilities, I used Softmax
    to transform the raw scores of the model into a probability distribution
    across classes.
    """
    all_probs: list = [];
    all_labels: list = [];

    scaled_model: TemperatureScaler = TemperatureScaler(model);
    temperature = scaled_model.set_temperature(val_loader, device); #is this val_loader is the same as the calibation_loader?
    print(f"Optimal T = {temperature}");
    scaled_model.eval();
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs: torch.Tensor = inputs.to(device);
            outputs: torch.Tensor = scaled_model(inputs);
            probs: torch.Tensor = functional.softmax(outputs, dim=1);
            probs: np.ndarray = probs.cpu().numpy();
            all_probs.append(probs);
            all_labels.append(labels.numpy());
    all_probs = np.concatenate(all_probs);
    all_labels = np.concatenate(all_labels);
    prob_pos = all_probs.max(axis=1); #highest predicted class
    ece = multiclass_ece(prob_pos, all_labels);
    #brier = brier_score_loss(all_labels, prob_pos);
    #plot_calibration(fop, mpv);
    # return (brier);
    return (0);



def main():
    device: torch.device = load_device();
    model: torch.nn.Module = load_model("best_model_state.pth", device);
    
    model.eval();
    val_loader: DataLoader = get_val_dataloader(64);
    brier = calibration_metrics(model, val_loader, device);
    print(f"brier dataype = {type(brier)}");

if __name__ == "__main__":
    main();
