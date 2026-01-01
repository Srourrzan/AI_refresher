import torch;
import torch.nn as nn;
from model import CNN;
from torch import Tensor, device;
import torch.optim as optim;
from torchvision import datasets;
from torch.utils.data import DataLoader;
from data_loader import get_stratified_and_calib_dataloaders, get_dataset;

from utils import load_device;

def train_one_epoch(model: nn.Module, dataloader: DataLoader, criterion, optimizer, device_: device) -> float:
    running_loss: float = 0.;

    model.train(); # this activates traning behaiour like dropout layers
    for inputs, labels in dataloader:
        inputs: Tensor = inputs.to(device_);
        labels: Tensor = labels.to(device_);
        optimizer.zero_grad(); #reset gradients from the previous step;
        outputs: Tensor = model(inputs);
        loss = criterion(outputs, labels);
        loss.backward();
        optimizer.step();
        running_loss += loss.item();
    return (running_loss / len(dataloader)); #return the average loss for the epoch;
        
def evaluate(model: nn.Module, dataloader: DataLoader, device_: device) -> float:
    correct: int = 0;
    total: int = len(dataloader.dataset);

    model.eval(); # this disables the activated layers in the training;
    with torch.no_grad(): # to stop the model from calculating gradients;
        for inputs, labels in dataloader:
            inputs = inputs.to(device_);
            labels = labels.to(device_);
            outputs = model(inputs);
            preds = outputs.argmax(1); # take the highest score;
            correct += (preds == labels).sum().item();
    return (correct / total);

def train_fold(train_dl: DataLoader, val_dl: DataLoader, device_: device, fold_id: int) -> nn.Module:
    best_val_acc: float = 0.0;
    best_state: dict = None;
    model: nn.Module = CNN().to(device_);
    criterion = nn.CrossEntropyLoss();
    optimizer = optim.Adam(model.parameters(), lr=1e-3);

    for epoch in range(20):
        train_loss: float = train_one_epoch(model, train_dl, criterion, optimizer, device_);
        val_acc: float = evaluate(model, val_dl, device_);
        print(f"Fold {fold_id} | Epoch {epoch} | Train loss: {train_loss:.4f}, Val accu: {val_acc:.4f}");
        if (val_acc > best_val_acc):
            best_val_acc = val_acc;
            best_state = model.state_dict().copy();
    print(f"best Val accu {best_val_acc:.4f}");
    return (best_state, best_val_acc);

def main():
    best_fold_if: int = -1;
    best_global_acc: float = 0.0;
    best_global_state: dict = None;
    dataset: datasets = get_dataset();
    folds, _ = get_stratified_and_calib_dataloaders(dataset);
    print(f"folds type {type(folds)}");
    save_path: str = "best_model_state.pth";
    device_: device = load_device();
    
    for fold_id, (train_dl, val_dl) in enumerate(folds):
        model_best_state, fold_best_acc = train_fold(train_dl, val_dl, device_, fold_id + 1);
        if (fold_best_acc > best_global_acc):
            best_global_acc = fold_best_acc;
            best_global_state = model_best_state;
            best_fold_id = fold_id + 1;
    torch.save(best_global_state, save_path);
    print(f"Best model from Fold {best_fold_id} with cv val acc = {best_global_acc:.4f} to {save_path}");
    return ;

if __name__ == "__main__":
    main();

#to store additional info about the model, I can do it like this:
# checkpoint = {
#    "model_state_dict": model.state_dict(),
#    "optimizer_state_dict": optimizer.state_dict(),
#    "epoch": epoch,
#    "val_loss": val_loss
#}
#torch.save(checkpoint, "checkpoint.pth")
