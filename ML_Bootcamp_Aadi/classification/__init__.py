import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path
sys.path.append(parent_dir)

from classification.logistic_regression import *
from classification.knn import *
from classification.ann import *
