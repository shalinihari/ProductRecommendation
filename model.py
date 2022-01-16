# import libraries
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
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
    
    def finditemRecommendation(self, user):
        name = [];
        flag = False
        tops=pd.read_pickle('./Top20Item2Classifier.pkl')
        raw_data = pd.read_csv("./sample30.csv")
        
        userList = raw_data[raw_data['reviews_username']==user][['id']]
        print("======userlist=====")
        print(userList)
        if userList.empty:
            print('DataFrame is empty!')
            return ""
        newlist = tops[user]
        
        print("========NewList============")
        print(newlist)
        second_tuple_elements = []
        for a_tuple in newlist:
            second_tuple_elements.append(a_tuple[0])
        print("=======Item recommended")
        print(second_tuple_elements)
        filter_top20_itemmod2 = raw_data[raw_data['id'].isin(second_tuple_elements)][['id','brand','categories','name','reviews_rating']]
        filter_top5_itemmod2 = filter_top20_itemmod2.groupby('id')['reviews_rating'].agg(["sum","mean", "count"]).reset_index().sort_values(by='mean', ascending=False).head()
        top5list_Itemmod2 = filter_top5_itemmod2['id'].to_list()
        top5Product_itemmodel2 = raw_data[raw_data['id'].isin(top5list_Itemmod2)][['id','name','brand','manufacturer','categories']].drop_duplicates()
        top5Product_itemmodel2['categories'] = top5Product_itemmodel2.categories.apply(lambda x: str(x.replace(',','')))
        suggestjson = top5Product_itemmodel2.to_json(orient='records')
        
        print(suggestjson)  
        flag = True
        if(flag == True):
            return suggestjson
        else:
            return ""
        return ""

        
    def purchaselist(self,user):
        raw_data = pd.read_csv("./sample30.csv")
        new_df = raw_data[raw_data['reviews_username']==user][['id','name','brand','manufacturer','categories']].drop_duplicates() 
        new_df['categories'] = new_df.categories.apply(lambda x: str(x.replace(',','')))
        print("======================== Newdf =================================================")
        print(new_df)
        purchasedjson = new_df.to_json(orient='records')
        print("=====================================================")
        print(purchasedjson)
        print("END=====================================================")
        return purchasedjson


    def finduserRecommendation(self, user):
        name = [];
        flag = False
        tops=pd.read_pickle('./top20Userrating.pkl')
        raw_data = pd.read_csv("./sample30.csv")

        userList = raw_data[raw_data['reviews_username']==user][['id']]
        print("======userlist=====")
        print(userList)
        if userList.empty:
            print('DataFrame is empty!')
            return ""

        newuserlist = tops[user]
        print("========NewList============")
        print(newuserlist)
        first_tuple_elements = []
        for a_tuple in newuserlist:
            first_tuple_elements.append(a_tuple[0])
        print("=======Item recommended")
        print(first_tuple_elements)
        filter_top20 = raw_data[raw_data['id'].isin(first_tuple_elements)][['id','brand','categories','name','reviews_rating']]
        filter_top5 = filter_top20.groupby('id')['reviews_rating'].agg(["sum","mean", "count"]).reset_index().sort_values(by='mean', ascending=False).head()
        top5list = filter_top5['id'].to_list()
        top5Product = raw_data[raw_data['id'].isin(top5list)][['id','name','brand','manufacturer','categories']].drop_duplicates()
        top5Product['categories'] = top5Product.categories.apply(lambda x: str(x.replace(',','')))
        suggestjson = top5Product.to_json(orient='records')
        flag = True
        if(flag == True):
            return suggestjson
        else:
            return ""
        return 
        

    
    def findSentiment(self, text):
        Result_sent = ""
        tfs=pd.read_pickle('./tfidfvec.pkl')
        logModel = pd.read_pickle('./LogisticClassifier_model.pkl')
        text=self.cleandata(text)
        print("Cleaned data is")
        print(text)
        text=self.lemmatizer(text)
        print("After Applying Lemmatization")
        print(text)
        Transformed = tfs.transform([text])
        print("After Vector transformation..")
        print(Transformed)
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
        sentence = [wordnet_lemmatizer.lemmatize(word) for word in word_tokenize(text)]
        return " ".join(sentence)

    def popularity(self):
        flag = False
        pop_df=pd.read_pickle('./PopularityModel.pkl')
        json = pop_df.to_json(orient='records')
        flag = True
        return json
        if(flag == False):
            return ""
        return 