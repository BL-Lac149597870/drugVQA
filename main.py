'''
Author: QHGG
Date: 2021-02-27 13:42:43
LastEditTime: 2021-03-01 23:26:38
LastEditors: QHGG
Description: 
FilePath: /drugVQA/main.py
'''
import torch
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")
torch.cuda.set_device(0)
print('cuda size == 1')
from trainAndTest import *
import time
def timeLable():
    return  time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    
def main():
    """
    Parsing command line parameters, reading data, fitting and scoring a SEAL-CI model.
    """
    losses,accs,testResults = train(trainArgs)
    with open("logs/"+ timeLable() +"losses.txt", "w") as f:
        f.writelines([str(log) + '\n' for log in losses])
    with open("logs/"+ timeLable() +"accs.txt", "w") as f:
        f.writelines([str(log) + '\n' for log in accs])
    with open("logs/"+ timeLable() +"testResults.txt", "w") as f:
        f.writelines([str(log) + '\n' for log in testResults])
    
if __name__ == "__main__":
    main()
