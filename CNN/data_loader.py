#import torch;
import numpy as np;
from torchvision import datasets;
from torchvision import transforms as T;
from torch.utils.data import DataLoader, Subset;
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit;

def get_dataset( ):
    transform: T.Compose = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))]);
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform);
    return (dataset);

def get_stratified_and_calib_dataloaders(dataset: datasets, n_folds=5, calib_fraction=0.2, batch_size=64) -> (list, DataLoader):
    """
    Split dataset into: CV part 80% and Calibration part 20%.
    """
    targets: list = dataset.targets;
    #split cv and calib
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=calib_fraction, random_state=42);
    zeros: np.ndarray = np.zeros(len(targets));
    cv_indices: np.ndarray;
    calib_indices: np.ndarray;
    cv_indices, calib_indices = next(splitter.split(zeros, targets));
    targets_np: np.array = np.array(targets);
    cv_targets: np.ndarray = targets_np[cv_indices];
    skf = StratifiedKFold(n_splits = n_folds, shuffle=True, random_state=42);
    folds: list = [];
    for train_idx, val_idx in skf.split(np.zeros(len(cv_indices)), cv_targets):
        fold_train_idx = cv_indices[train_idx];
        fold_val_idx = cv_indices[val_idx];
        train_dl = DataLoader(Subset(dataset, fold_train_idx), batch_size=batch_size, shuffle=True);
        val_dl = DataLoader(Subset(dataset, fold_val_idx), batch_size=batch_size, shuffle=True);
        folds.append((train_dl, val_dl));
    calib_dl = DataLoader(Subset(dataset, calib_indices), batch_size=batch_size, shuffle=False);
    print(f"CV folds: {len(folds)}");
    print(f"Calib set: {len(calib_indices)}");
    return (folds, calib_dl);

def get_test_dataloader(batch_size: int = 64) -> DataLoader:
    transform: T.Compose = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,))
    ]);
    test_dataset: datasets = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform);
    dl: DataLoader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False);
    return (dl);

# if __name__ == "__main__":
#     data: datasets = get_dataset();
#     # dl = get_stratified_dataloaders(data);
#     get_stratified_and_calib_dataloaders(data);
