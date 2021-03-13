'''
Author: QHGG
Date: 2021-03-03 21:54:59
LastEditTime: 2021-03-03 22:41:27
LastEditors: QHGG
Description: 
FilePath: /drugVQA/map.py
'''
from utils import *
import numpy as np
seq, contactMap = getProtein('./data/DUDE/contactMap', 'aa2ar_3emlA_full', True)
contactmap_np = [list(map(float, x.strip(' ').split(' '))) for x in contactMap]
feature2D = np.expand_dims(contactmap_np, axis=0)
feature2D = torch.FloatTensor(feature2D).unsqueeze(0)
