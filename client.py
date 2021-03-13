'''
Author: QHGG
Date: 2021-03-04 16:04:16
LastEditTime: 2021-03-04 16:49:59
LastEditors: QHGG
Description: 
FilePath: /drugVQA/client.py
'''
import requests

payload = {"smi": "CNS(=O)(=O)c1ccc2c(c1)CCN2S(=O)(=O)c3cc(ccc3C(=O)OC)C(=O)OC"}

resp = requests.post("http://localhost:5000/predict", data=payload)
print(resp.json()['pred'])