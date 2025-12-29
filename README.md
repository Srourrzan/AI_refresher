# AI Refresher with PyTorch

A focused repository for learning Artificial Intelligence fundamentals with PyTorch, covering Artificial Neural Networks (ANN), Convolutional Neural Networks (CNN), and Optical Character Recognition (OCR).

## 📚 Overview

This refresher covers practical implementations and in-depth explanations of:

- **Artificial Neural Networks (ANN)** - Fundamentals, backpropagation, and training strategies
- **Convolutional Neural Networks (CNN)** - Image processing, feature extraction, and classification
- **Optical Character Recognition (OCR)** - Text detection and recognition using deep learning

## 🛠️ Environment

This repository is designed to be used with **Google Colab** for easy access to GPU/TPU resources without local installation.

## 📖 Usage

Each section contains:
- **Theory explanations** with key concepts
- **Code examples** demonstrating implementations
- **Exercises** for hands-on practice
- **Jupyter notebooks** for interactive exploration

To use with Google Colab:

```python
# Clone the repository in Colab
!git clone https://github.com/Srourrzan/AI_refresher.git

# Install any additional dependencies (if needed)
!pip install torch torchvision
```

Then open the notebooks in Colab for interactive learning with GPU acceleration.


## 🎯 Learning Path

1. **Artificial Neural Networks (ANN)**
   - Tensors and PyTorch fundamentals
   - Building simple neural networks
   - Activation functions and loss functions
   - Training and optimization

2. **Convolutional Neural Networks (CNN)**
   - Convolution and pooling operations
   - Building CNN architectures
   - Image classification tasks
   - Feature visualization

3. **Optical Character Recognition (OCR)**
   - Text detection in images
   - Character recognition models
   - End-to-end OCR systems
   - Practical applications


### Fine-Tuning Neural Network Hyperparameters

#### Number of Hidden Layers
For many problems, you can begin with a single hidden layer and get reasonable results. A NN with just one hidden layer can theoretically model 
even the most complex functions, provided it has enough neurons. But for complex problem, deep networks have a much higher parameter efficiency
than shallow ones: thay can model complex functions using exponentially fewer neurons than shallow nets, allowing them to reach much better 
performance with the same amount of training data.

#### Number of Neurons per Hidden Layer
As for the hidden layers, it used to be common to size them to form a pyramid, with fewer and fewer neurons at each layer-the rational being that
many low-level features can coalesce into far fewer high-level features. However, this practice has been largely abandoned because it seems that using the same number of neurons in all hidden layers performs just as well in most cases, or even better. That said, depending on the dataset, it can sometimes help to make the first hidden layer bigger than others.

One can try increasing the number of neurons gradually until the network starts overfitting. Alternatively, you can try building a model with a bit more layers and neurons than you actually need, then use early stopping and other regularization techniques to prevent it from overfitting.

#### Learning Rate
The learning rate is the most important hyperparameter. The optimal learning rate is about half of the maximum learning rate (the learning rate above which the training algorithm diverges). One way to find a good learining rate is to train the model for a few hundred iterations, starting with a very low learning rate (e.g., 10^{-5}) and gradually increasing it up to a very large value (e.g., 10). This is done by multiplying the learning rate by a constant factor at each iteration (e.g., by (10 / 10^{-5))^{1/500} to go from 10^{-5} to 10 in 500 iterations).
Plot the loss as a function of the learning rate (using a log scale for the learning rate), you should see it dropping at first. But after a while, the learning rate will be too large, so the loss will shoot back up: the optimal learning rate will be a bit lower than the point at which the loss starts to climb (typically about 10 times lower than the turning point). (or better use learning rate scheduling)

#### Optimizer
Choose a better optimizer than Mini-batch Gradient Descent.

#### Batch Size
One strategy is to try to use a large batch size, using learning rate warmup, and if training is unstable or the final performance is disappointing, then try using a small batch size instead.

#### Activation function
In general, the ReLU (or better, use leaky ReLU) activation function is a good default for all hidden layers. For the output layer, it depends on the task. ELU, GELU, and Swish are better.


Glorot and Bengio proposed a good compromise that has proven to work
very well in practice: the connection weights of each layer must be
initialized randomly as described in Equation 11-1, where fan = (fan +
fan )/2. This initialization strategy is called Xavier initialization or Glorot
initialization, after the paper’s first author.

Normal distribution with mean 0 and variance σ2 = 1
fanavg
Or a uniform distribution between − r and + r, with r = √ 3
fanavg
Using Glorot initialization can speed up training considerably, and
it is one of the tricks that led to the success of Deep Learning.

#### Batch Normalization
This technique consists of adding an operation in the model just before or after the activation function of each hidden layer. This operation simply zero-center and normalizes each point, then scales and shifts the result using two new parameters vectors per layer: one for scaling, the other for shifting (The operation lets the model learn the optimal scale and mean of each of the layer's inputs).

### What is Calibration

**Calibration** makes sure that a model's estimated probabilities match real-world likelihood. For example, if a weather forecasting model predicts a 70% chance of rain on several days, then roughly 70% of those days should actually be rainy for the model to be considered well calibrated. This makes model predictions more reliable and trustworthy.

![f1_reliability_diafram.png](f1_reliability_diafram.png)
<sub>If the prediction line is under the reliability line then the model is predicting the true probability, and if the prediction line is above the reliability line, then the model is under-predicting the true probability.</sub>

Log-loss (or cross entropy) penalises models that are too overconfident when making wrong predictions or making predictions that differ significantly from their true probabilities.

## 🔗 Resources

- [PyTorch Official Documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Deep Learning with PyTorch](https://pytorch.org/assets/deep-learning/Deep-Learning-with-PyTorch.pdf)
- [understanding-model-calibration-a-gentle-introduction-visual-exploration](https://towardsdatascience.com/understanding-model-calibration-a-gentle-introduction-visual-exploration/)
- [Understanding Model Calibration](https://arxiv.org/html/2501.19047v1)