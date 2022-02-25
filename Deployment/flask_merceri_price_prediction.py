import pandas as pd
import numpy as np
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import pickle
from flask import Flask, jsonify, request
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import tensorflow as tf
from tensorflow.keras import Sequential,layers,Input,optimizers,callbacks,losses,Model
from tensorflow.keras.layers import Dense,Embedding,Flatten,Dropout,BatchNormalization,Concatenate,LSTM
from tensorflow.keras.losses import mean_squared_error
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import plot_model
import tensorflow.keras.backend as k
from tensorflow.keras.models import load_model
from bs4 import BeautifulSoup
import re
import flask

app = Flask(__name__,template_folder='template')

#----Preprocess Functions---
def decontracted(phrase):
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def text_preprocess(phrase):

    stop_words = set(stopwords.words('english'))
    words = list()
    phrase = decontracted(phrase)
    for word in phrase.split(' '):
        if word not in stop_words:
          word = word.replace(r'[^A-Za-z0-9]','')
          '''for char in string.punctuation:
            word= word.replace(char,'')'''
          word = re.sub(r'[^\w\s]', '', word) ### Ref: https://www.geeksforgeeks.org/python-remove-punctuation-from-string/
          word = word.lower()
          words.append(word)
    if len(words)==0:
        words.append("missing")
    phrase = ' '.join(words)
    return phrase


def item_cat_transform(x):
    cat_dict = {1:'New',2:'Like New',3:'Good',4:'Fair',5:'Poor'}
    return cat_dict[x]

def text_test_token(test_docs,feature,maxlength):
    filename = './token_'+feature+'.pkl'
    with open(filename,'rb') as out_file:
        t = pickle.load(out_file)
    vocab_size = len(t.word_index) + 1
    # integer encode the documents
    encoded_docs = t.texts_to_sequences(test_docs)
    padded_docs = pad_sequences(encoded_docs, maxlen=maxlength, padding='pre')

    return padded_docs

def encode(testdata,feature,encodetype):
    if encodetype == 'CountVectorizer':
        filename = './Countvectorizer_'+feature+'.pkl'
        with open(filename,'rb') as out_file:
           CountVectorizer = pickle.load(out_file)
        data = CountVectorizer.transform(testdata)
    elif encodetype == 'OneHotEncoder':
        filename = './OneHotEncoder_'+feature+'.pkl'
        with open(filename,'rb') as out_file:
           OneHotEncoder = pickle.load(out_file)
        data = OneHotEncoder.transform(testdata)
    elif encodetype == 'MinMaxScaler':
        filename = './MinMaxScaler_'+feature+'.pkl'
        with open(filename,'rb') as out_file:
            MinMaxScaler = pickle.load(out_file)
        data = MinMaxScaler.transform(testdata)
        
    return data
def embed_preprocess(testdata):

    x_test_item_cnd = encode(testdata.item_condition_id.values,'item_condition_id','CountVectorizer')
    x_test_shipping = encode(testdata.shipping.values.reshape(-1,1),'shipping','OneHotEncoder')
    x_test_len_words_name = encode(testdata.len_words_name.values.reshape(-1,1),'len_words_name','MinMaxScaler')
    x_test_len_words_item_description = encode(testdata.len_words_item_description.values.reshape(-1,1),'len_words_item_description',\
                                                 'MinMaxScaler')

    brand_name_padded_docs_train = text_test_token(testdata['brand_name'].values,'brand_name',6)
    Tier1_padded_docs_train = text_test_token(testdata['Tier1_category_name'].values,'Tier1_category_name',3)
    Tier2_padded_docs_train = text_test_token(testdata['Tier2_category_name'].values,'Tier2_category_name',5)
    Tier3_padded_docs_train = text_test_token(testdata['Tier3_category_name'].values,'Tier3_category_name',7)
    name_padded_docs_train = text_test_token(testdata['name'].values,'name',17)
    desc_padded_docs_train = text_test_token(testdata['item_description'].values,'item_description',235)

    test_data = [x_test_item_cnd.toarray(),x_test_shipping.toarray(),brand_name_padded_docs_train,Tier1_padded_docs_train,\
                Tier2_padded_docs_train,Tier3_padded_docs_train,name_padded_docs_train,desc_padded_docs_train,x_test_len_words_name,\
                    x_test_len_words_item_description]

    return test_data

#--DF Transform--
def data_preprocess(data):

    data['name'] = data['name'].apply(lambda x: text_preprocess(x))
    data['item_description'] = data['item_description'].apply(lambda x: text_preprocess(x))
    data['shipping'] = data['shipping'].astype('int64')
    data['item_condition_id'] = data.item_condition_id.apply(lambda x: item_cat_transform(int(x)))
    data['len_words_name']= data.name.apply(lambda x: len(x.split(' ')))
    data['len_words_item_description']= data.item_description.apply(lambda x: len(x.split(' ')))
    data['Tier1_category_name'] = data.category_name.apply(lambda x: x.split('/')[0].lower() if len(x.split('/')) > 0 else x)
    data['Tier2_category_name'] = data.category_name.apply(lambda x: x.split('/')[1].lower() if len(x.split('/')) > 1 else x)
    data['Tier3_category_name'] = data.category_name.apply(lambda x: x.split('/')[2].lower() if len(x.split('/')) > 2 else x)
    data.drop(columns = 'category_name', inplace = True)
    
    return data

#--Performance Metric Function call--
def root_mean_squared_log_error(y_true, y_pred):
    return k.sqrt(mean_squared_error(y_true, y_pred))


#--API Call--
@app.route('/')
def welcome():
    return 'Welcome to Merceri Price Suggestion!'


@app.route('/price_pred')
def pred():
    return flask.render_template('Merceri_Price_Pred.html')

@app.route('/predict', methods=['POST'])
def predict():
    print(request.form)
    to_predict_list = request.form.to_dict()
    for key in to_predict_list.keys():
      if to_predict_list[key] == '' or to_predict_list[key] == None or to_predict_list[key] == 'None':
        if key == 'item_description':
          to_predict_list[key] = 'no description yet'
        elif key == 'Product Name':
          to_predict_list[key] = 'no title yet'
        else:
          to_predict_list[key] = 'missing'
  
    df = pd.DataFrame(to_predict_list,index=[0])
    preprocess_df=data_preprocess(df)
    model = load_model('./Final_model_4.h5',custom_objects={'root_mean_squared_log_error':root_mean_squared_log_error})
    data = embed_preprocess(preprocess_df)
    predicted_price = model.predict(data)
    print(predicted_price[0].item())

    return jsonify({'Predicted Sales Price': np.exp(predicted_price[0].item()).round(2)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)