# import libraties
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
#--- HTML Tag Removal
import re 
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize



class Recommendation:
    
    def __init__(self):
        nltk.data.path.append('./nltk_data/')
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        lemmatizer = nltk.stem.WordNetLemmatizer()
        self.wordnet_lemmatizer = WordNetLemmatizer()
                
    def chkUserList(self, user):
        raw_data = pd.read_csv("./sample30.csv")
        user_df = raw_data[['reviews_username']]
        user_df = user_df.drop_duplicates()
        if not user_df['id'].isin(user):
            print("username not found")
            return "User not Found"
        return

    def finduserRecommendation(self, user):
        name = [];
        flag = False
        tops=pd.read_pickle('./top5Classifier.pkl')
        raw_data = pd.read_csv("./sample30.csv")
        
        for uid, user_rating in tops.items():
            if uid==user:
                itemList = [iid for (iid,rating) in user_rating]
                print(itemList)
                test_df = raw_data[['id','name','manufacturer','brand']]
                test_df = test_df.drop_duplicates()
                json = test_df[test_df['id'].isin(itemList)].to_json(orient='records')
                # name = test_df[test_df['id'].isin(itemList)].name.values.tolist()
                # lstname = ','.join([str(elm) for elm in name])
                flag = True
                return json
        if(flag == False):
            return ""
        return 

    
    def findSentiment(self, text):
        Result_sent = ""
        tfs=pd.read_pickle('./tfidfvec.pkl')
        naivebys=pd.read_pickle('./naiveBayes_model.pkl')
        logModel = pd.read_pickle('./LogisticClassifier_model.pkl')
        stop_words = stopwords.words('english')
        text=self.cleandata(text)
        print("Cleaned data is")
        print(text)
        text=self.lemmatizer(text)
        print("After Applying Lemmatization")
        print(text)
        Transformed = tfs.transform([text])
        print("After Vector transformation..")
        print(Transformed)
        PredictText = naivebys.predict(Transformed)
        print("After Naivebayes step")
        print(PredictText[0])
        if PredictText[0] == 0:
            Result_sent = "Negative"
        else:
            Result_sent = "Positive"
        PredictLogText = logModel.predict(Transformed)
        print("After Logistic Model step")
        print(PredictLogText[0])
        if PredictLogText[0] == 0:
            LogResult_sent = "Negative"
        else:
            LogResult_sent = "Positive"
        return LogResult_sent
    
    def cleandata(self,text):
        text = text.lower()
        text = re.sub("[\(\[].*?[\)\]]", "", text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub('[^a-zA-Z]', ' ', text)    
        return text
    
    def lemmatizer(self, text):
        wordnet_lemmatizer = WordNetLemmatizer()   
        stop_words = stopwords.words('english')     
        sentence = [wordnet_lemmatizer.lemmatize(word) for word in word_tokenize(text) if not word in stop_words]
        return " ".join(sentence)
