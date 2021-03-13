'''
Author: QHGG
Date: 2021-03-12 21:33:00
LastEditTime: 2021-03-12 22:20:28
LastEditors: QHGG
Description: 
FilePath: /drugVQA/mcnet.py
'''
'''
Author: QHGG
Date: 2021-03-12 21:41:32
LastEditTime: 2021-03-12 21:52:36
LastEditors: QHGG
Description: 
FilePath: /my-app/app.py
'''
from flask import Flask, jsonify, request
from flask_cors import CORS
import json
app = Flask(__name__)
CORS(app, resources=r'/*')

@app.route('/', methods=['GET'])
def predict():
    if request.method == 'GET':
        
        return jsonify({1:1})


    
if __name__ == "__main__":
    app.run('0.0.0.0', port='5001')

