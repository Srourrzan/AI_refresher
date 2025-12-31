import torch;
import torch.nn as nn;
from model import CNN;
import torch.optim as optim;
from torch.utils.data import DataLoader;
from data_loader import get_stratified_dataloaders, get_dataset;


def train_one_epoch(model: nn, dataloader: DataLoader, criterion, optimizer, device) -> float:
    running_loss: float = 0.;

    model.train(); # this activates traning behaiour like dropout layers
    for inputs, labels in dataloader:
        inputs = inputs.to(device);
        labels = labels.to(device);
        optimizer.zero_grad(); #reset gradients from the previous step;
        outputs = model(inputs);
        loss = criterion(outputs, labels);
        loss.backward();
        optimizer.step();
        running_loss += loss.item();
    return (running_loss / len(dataloader)); #return the average loss for the epoch;
        
def evaluate(model: nn, dataloader: DataLoader, device) -> float:
    correct: int = 0;

    model.eval(); # this disables the activated layers in the training;
    with torch.no_grad(): # to stop the model from calculating gradients;
        for inputs, labels in dataloader:
            inputs = inputs.to(device);
            labels = labels.to(device);
            outputs = model(inputs);
            preds = outputs.argmax(1); # take the highest score;
            correct += (preds == labels).sum().item();
    return (correct / len(dataloader.dataset));


def run_training(train_dl, val_dl, device) -> nn:
    model = CNN().to(device);
    criterion = nn.CrossEntropyLoss();
    optimizer = optim.Adam(model.parameters(), lr=1e-3);

    for epoch in range(10):
        train_loss = train_one_epoch(model, train_dl, criterion, optimizer, device);
        val_acc = evaluate(model, val_dl, device);
        print(f"Epoch {epoch} | Train loss: {train_loss:.4f}, Val accu: {val_acc:.4f}");
    return (model);


def main():
    dataset = get_dataset();
    folds = get_stratified_dataloaders(dataset);
    save_path = "best_model_state.pth";
    
    if (torch.cuda.is_available()):
        device = torch.device("cuda");
    else:
        device = torch.device("cpu");
    for train_dl, val_dl in folds:
        model = run_training(train_dl, val_dl, device);
    torch.save(model.state_dict(), save_path);
    print(f"Model weights saved to {save_path}");

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
