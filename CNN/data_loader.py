#import torch;
from torchvision import datasets;
from torchvision import transforms as T;
from sklearn.model_selection import StratifiedKFold;
from torch.utils.data import DataLoader, Subset;

def get_dataset( ):
    transform = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))]);
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform);
    return (dataset);

def get_stratified_dataloaders(dataset:datasets, n_splits: int = 5, batch_size: int = 64) -> list:
    targets = dataset.targets;
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42);
    folds = [];
    
    for train_idx, val_idx in skf.split(dataset.data, targets):
        train_dl = DataLoader(Subset(dataset, train_idx), batch_size=batch_size);
        val_dl = DataLoader(Subset(dataset, val_idx), batch_size=batch_size);
        folds.append((train_dl, val_dl));
    return (folds);

if __name__ == "__main__":
    data = get_dataset();
    print(f"data type = {type(data)}");
    folds = get_stratified_dataloaders(data);
    print(f"folds datatype = {type(folds)}");