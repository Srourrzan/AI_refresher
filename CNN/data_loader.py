#import torch;
from torchvision import datasets;
from torchvision import transforms as T;
from torch.utils.data import DataLoader, Subset;
from sklearn.model_selection import StratifiedKFold;

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

def get_val_dataloader(batch_size: int) -> DataLoader:
    transform = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))]);
    val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform);
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False);
    return (val_loader);

