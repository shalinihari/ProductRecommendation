

# This is the engine of the REST based flask API

from flask import Flask, render_template, request, redirect, url_for, jsonify, abort
import pickle

from nltk.sem.evaluate import Error
from model import Recommendation

recommend = Recommendation()
app = Flask(__name__)  # intitialize the flaks app  # common 

import os
from flask import send_from_directory


@app.route('/pr.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'pr.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/', methods = ['POST', 'GET'])
def home():
    flag = False 
    if request.method =='POST':
        flag = True
        user = request.form["userid"]
        print("The user is "+ user)
        data=recommend.findSentiment(user) 
        return render_template("index.html", placeholder_text = data)
    if request.method =='GET':
        return render_template("index.html", placeholder_text = "Hello fro GET method")
    return render_template("index.html")
    

@app.route('/userList', methods = ['GET'])
def userList():
    data=recommend.getUsers()
    return data

@app.route('/SentimentAnalysis', methods = ['POST'])
def SentimentAnalysis():
    flag = True
    sentText = request.args.get("param1")
    print(sentText)
    data=recommend.findSentiment(sentText)  
    print(data)  
    # return render_template('index.html', placeholder_text=data)
    return data

@app.route('/UserRecommendation', methods = ['POST'])
def UserRecommendation():
    try:
        sentText = request.args.get("param1")
        print(sentText)
        data=recommend.finduserRecommendation(sentText)  
        print("=====================")
        print(data) 
        return data
    except Error:
        abort (500)

@app.errorhandler(404)
def page_not_found(error):
    return render_template('index.html'), 404

@app.route('/productList', methods = ['GET'])
def productList():
    user=request.args.get("userid")
    data=recommend.getTopProductsNew(user)
    return data

if __name__ == '__main__' :
    app.run(debug=True )  
    
    #,host="0.0.0.0")
