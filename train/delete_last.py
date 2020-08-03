#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import os
import numpy as np

from_data = np.load("train/train.npy")
print from_data
from_data = np.delete(from_data, (np.size(from_data, 0)-1), axis = 0)
print from_data
np.save("train/train", from_data)
