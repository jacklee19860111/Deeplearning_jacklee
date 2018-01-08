import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import load_digits
from sklearn.preprocessing import LabelBinarizer
from sklearn.cross_validation import train_test_split
from NeuralNetwork import NeuralNetwork

digits = load_digits()
X = digits.data
y = digits.target
X -= X.min()
X /= X.max()

 
