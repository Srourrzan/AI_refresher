import torch;
import numpy as np;
import matplotlib.pyplot as plt;

def imshow(img: torch.Tensor) -> None:
    if not isinstance(img, torch.Tensor):
        raise TypeError("img must be a torch.Tensor");
    img = img / 2 + 0.5;
    npimg = img.numpy();
    plt.show(np.transpose(npimg, (1, 2, 0)));
    plt.show();

if __name__ == "__main__":
    imshow("str");