import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
from torchtext import data, datasets
import matplotlib.pyplot as plt
import seaborn
seaborn.set_context(context="talk")
#%matplotlib inline
