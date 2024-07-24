# MNIST Dataset Neural Network

This project demonstrates how to build and train a simple neural network from scratch to classify digits from the MNIST dataset. The MNIST dataset consists of 28x28 pixel grayscale images of handwritten digits (0-9).

Dependancies used: Numpy, Pandas, Matplotlib

```
pip install numpy pandas matplotlib
```

# Neural Netowrk's Architecture

Input Layer: 784 neurons (28x28 pixels)
Hidden Layer 1: 128 neurons
Hidden Layer 2: 64 neurons
Output Layer: 10 neurons (one for each digit 0-9)

# Usage

Get the Data:
Download the MNIST Dataset (Also works on Fashion MNIST Dataset)
1. name the training dataset as "train.csv"
2. name the testing dataset as "test.csv"

Run the Script:
run

```
python Neural_Network.py
```

View Predictions:
The script will output the predictions and actual labels for some test samples, and display the corresponding images.
