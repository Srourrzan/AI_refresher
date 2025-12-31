import numpy as np;
import matplotlib.pyplot as plt;
from torchvision import datasets;
from torch.utils.data import DataLoader;
from sklearn.metrics import brier_score_loss;
from torch import nn, device, Tensor, from_numpy, no_grad;
from torchmetrics.classification import CalibrationError;
from sklearn.calibration import calibration_curve, CalibratedClassifierCV;

from model import CNN;
from utils import load_model, load_device;
from temperature_model import TemperatureScaler;
from data_loader import get_calibration_dataloader, get_dataset;

def plot_calibration(frac_of_postv: np.ndarray, mean_pred_val: np.ndarray, bin_counts: np.ndarray=None):
    plt.figure(figsize=(8, 6));
    plt.plot(mean_pred_val, frac_of_postv, marker='o', label='Calibrated Model');
    plt.plot([0, 1], [0, 1], linestyle='--', color="gray", label="Perfectly calibrated");

    if bin_counts is not None:
        for i in range(len(mean_pred_val)):
            plt.text(mean_pred_val[i], frac_of_postv[i], f"{bin_counts[i]}",
                     ha="center", va="center", color="black",
                     fontsize=8);
    
    plt.title("Calibration Curve (Reliability Diagram)");
    plt.xlabel("Mean Predicted Value (Confidence)");
    plt.ylabel("Fraction of Positives (Accuracy)");
    plt.legend();
    plt.xlim(0, 1);
    plt.ylim(0, 1);
    plt.grid(alpha=0.3);
    plt.savefig("calibration_curve.png", dpi=300);
    return ;

def calculate_ece(task_type: str, probabilities: Tensor, labels: Tensor, num_classes: int, n_bins: int, device_:device) -> Tensor:
    ece: MulticlassCalibrationError = CalibrationError(
        task=task_type, num_classes=num_classes, n_bins=n_bins
    ).to(device_);
    ece_value: Tensor = ece(probabilities, labels).item();
    return (ece_value);
    
def calibration_metrics(model: nn.Module, cal_loader: DataLoader, device_: device) -> (Tensor, float, float):
    """
    To make the output of the model represent probabilities, I used Softmax
    to transform the raw scores of the model into a probability distribution
    across classes.
    """
    all_probs: list = [];
    all_labels: list = [];

    scaled_model: TemperatureScaler = TemperatureScaler(model);
    temperature: float = scaled_model.set_temperature(cal_loader, device_);
    print(f"Optimal T = {temperature}, type {type(temperature)}");
    scaled_model.eval();
    with no_grad():
        for inputs, labels in cal_loader:
            inputs: Tensor = inputs.to(device_);
            outputs: Tensor = scaled_model(inputs);
            probs: Tensor = nn.functional.softmax(outputs, dim=1);
            all_probs.append(probs.cpu().numpy());
            all_labels.append(labels.numpy());
    all_probs: np.ndarray = np.concatenate(all_probs);
    all_labels: np.ndarray = np.concatenate(all_labels);
    # ece expects the probability distribution [N, C] and labels [N]
    probs_tensor: Tensor = from_numpy(all_probs).to(device_);
    labels_tensor: Tensor = from_numpy(all_labels).to(device_);
    ece_value: Tensor = calculate_ece("multiclass", probs_tensor, labels_tensor, 10, 15, device_);
    print(f"ece_value {type(ece_value)} =\n{ece_value}");
    #calculate nll
    nll: float = nn.functional.cross_entropy(probs_tensor, labels_tensor).item();
    print(f"nll {type(nll)}=\n{nll}");
    #calculate accuracy
    predicted_classes: np.ndarray = np.argmax(all_probs, axis=1);
    print(f"predicted_classes {type(predicted_classes)}=\n{predicted_classes}");
    accuracy: np.float64 = np.mean(predicted_classes == all_labels);
    print(f"accuracy {type(accuracy)}=\n{accuracy}");
    # evaluate fop, mpv
    confidence: np.ndarray = np.max(all_probs, axis=1);
    print(f"confidence {type(confidence)}, =\n{confidence}");
    correct: np.ndarray = (predicted_classes == all_labels).astype(int);
    print(f"correct {type(correct)}=\n{correct}");
    fop: np.ndarray;
    mpv: np.ndarray;
    fop, mpv = calibration_curve(correct, confidence, n_bins=15, strategy='uniform');
    print(f"fop {type(fop)}=\n{fop}");
    print(f"mpv {type(mpv)}=\n{mpv}");
    bin_counts: np.ndarray = np.histogram(confidence, bins=15, range=(0, 1))[0];
    print(f"bin_counts {type(bin_counts)}=\n{bin_counts}");
    plot_calibration(fop, mpv, bin_counts);
    return (ece_value, nll, accuracy);



def main():
    device: torch.device = load_device();
    model: torch.nn.Module = load_model("best_model_state.pth", device);
    
    model.eval();
    train_data: datasets = get_dataset();
    calibration_loader: DataLoader = get_calibration_dataloader(train_data, 64);
    ece_value, nll, accuracy  = calibration_metrics(model, calibration_loader, device);

if __name__ == "__main__":
    main();
