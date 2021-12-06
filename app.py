

# Import Libraries ที่เกี่ยวกับ การจัดการ Deployment Streamlit
import streamlit as st
import time

# Libraries ที่เกี่ยวกับ การจัดการ สร้างตาราง และ Plot Graph

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

# Libraries ที่เกี่ยวกับ การจัดการ สร้าง Machine Learning Model และ Deep Learning Model
import pickle


import tensorflow as tf
from tensorflow.keras.models import load_model   # load saved model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences   # to do padding or truncating

#from tensorflow.keras.utils import np_utils
from tensorflow.keras.models import Sequential     # the model
from tensorflow.keras.layers import Embedding


# Libraries ที่เกี่ยวกับ การจัดการข้อมูล Text 

import re
from bs4 import BeautifulSoup  


import pythainlp
from pythainlp import sent_tokenize, word_tokenize
from pythainlp import word_tokenize, Tokenizer
from pythainlp.tokenize import Tokenizer
from pythainlp.corpus.common import thai_words
from pythainlp.util import dict_trie


# สร้าง Path สำหรับบันทึกข้อมูลต่าง 


filenames = "Equipment_and_Word_with__delete_Duplicate_rev02.csv"

import csv

with open(filenames, newline='') as f:
    reader = csv.reader(f)
    DATA_WORDS = list(reader)

Word_Data = pd.DataFrame(data = DATA_WORDS, columns=['Equipment'])

Word_Data['Equipment'] = Word_Data['Equipment'].astype('str')

Equipment_Word = Word_Data['Equipment'].tolist()


#เพิ่มคำใน Dictionary
custom_words_list = set(thai_words())
## add multiple words
custom_words_list.update(Equipment_Word)
## add word
# custom_words_list.add("ประเมิณ", "ประเมิน", "ความเสี่ยง", "รางระบาย", "ผู้ปฎิบัติงาน")
# custom_words_list.add("ประเมิณ", "ประเมิน", "ความเสี่ยง", "รางระบาย", "ผู้ปฎิบัติงาน")
trie = dict_trie(dict_source=custom_words_list)

# ดำเนินการ เพิ่มคำว่า "ประเมิณ", "ประเมิน", "ความเสี่ยง", "รางระบาย", "ผู้ปฎิบัติงาน" ใน engine= 'newmm'
custom_tokenizer1 = Tokenizer(custom_dict=trie, engine='newmm')

# ดำเนินการ เพิ่มคำว่า "ประเมิณ", "ประเมิน", "ความเสี่ยง", "รางระบาย", "ผู้ปฎิบัติงาน" ใน engine= 'multi_cut'
custom_tokenizer2 = Tokenizer(custom_dict=trie, engine="multi_cut")

# loading
with open('token04_Model1.pickle', 'rb') as handle:
    Load_Token = pickle.load(handle)

loaded_model = load_model('LSTM04_Model1.h5')


def Text_Prediction_Completion(Text_Prediction,  Tokenizer_Model, Prediction_Model) :
    
    maxlen = 100 
    Text_Original = Text_Prediction
    
    
    # แปลงข้อมูล List ให้เป็น Dataframe ก่อน เพื่อเป็น Input ให้แก่ Clean Function
    Text_df = pd.DataFrame(Text_Prediction, columns=["Text"])
    
    import re

    def clean(x):
        x = x.replace("\n", " ")
        x = x.replace("_x000d_", " ")
        x = re.sub(r'https?:\/\/.*[\r\n]*', '', x)
        # ลบสัญลักษณ์แปลก ออกไป
        x = re.sub(r'[!,;,^,ํ]', '', x)
        return x

    
    
    Text_df["Text_Clean"] = Text_df["Text"].apply(clean)

    # a new column ที่ Clean เครื่องหมาย \n ออกจาก Full Corpus
    Text_df["Text_Clean"] = Text_df["Text_Clean"].str.replace(r'\\n',' ', regex=True)

    # a new column ที่ Clean เครื่องหมาย _x000d_ ออกจาก Full Corpus 
    Text_df["Text_Clean"] = Text_df["Text_Clean"].str.replace(r'\_x000d_', ' ', regex=True)

    
    # แปลงข้อมูล Dataframe ให้เป็น  List เหมือนเดิม ก่อนเป็น Input ให้แก่ Tokenization_Embedding_Text Function ต่อไป
    Text_Clean = Text_df["Text_Clean"].tolist()
    
    #กำหนดให้เป็นตัวพิมพ์เล็ก
    Text_Clean_lowwercase = list(pd.Series(Text_Clean).str.lower())

    
    
    
    def Tokenization_Embedding_Text(Text_Clean, Load_Tokenizer) :
        Text_Final = []
        for i in range(len(Text_Clean)) :
            Text_Sequence = ' '.join(custom_tokenizer1.word_tokenize(Text_Clean[i]))
            Text_Final.append(Text_Sequence)
            
            Tokenize_Words = Load_Tokenizer.texts_to_sequences(Text_Final)
            Tokenize_Words = pad_sequences(Tokenize_Words, maxlen=maxlen)
            
        
        
        return Text_Final, Tokenize_Words

    
    Text_Final, Tokenize_Words = Tokenization_Embedding_Text(Text_Clean = Text_Clean_lowwercase, Load_Tokenizer = Tokenizer_Model)
    
    def Prediction_Text(Text_Embed_Final, Prediction_Model) :
        Text_prob_predict = Prediction_Model.predict(Text_Embed_Final)
        Text_predict = tf.squeeze(tf.round(Text_prob_predict))
        
        if Text_predict == 1 :
            Result_Predict = "Incident"
            Percent_confident = np.around((Text_prob_predict[0][0]*100), decimals=2)
        else :
            Result_Predict = "SWO"
            Percent_confident = np.around(((1-Text_prob_predict[0][0])*100), decimals=2)
            
        return Text_Final, Text_predict, Result_Predict , Text_prob_predict, Percent_confident   

    Text_Final, Text_predict, Result_Predict , Text_prob_predict, Percent_confident = Prediction_Text(Text_Embed_Final = Tokenize_Words, Prediction_Model = Prediction_Model)
    
    
    if Text_predict == 1 :
        print(f"Probability of Prediction : {Text_prob_predict[0][0]}")
        print(f"% Percent of Confident : {Percent_confident}"," %")
        print(f"Prediction : {Text_predict}", "(Incident)")
        print("\n")
        print(f"Original Text:\n{Text_Original}\n")
        print(f"Text:\n{Text_Final}\n")
        print("----------------------------------------------\n")
        Result_Predict = "Incident"
    else :
        print(f"Probability of Prediction : {Text_prob_predict[0][0]}")
        print(f"% Percent of Confident : {Percent_confident}"," %")
        print(f"Prediction : {Text_predict}", "(SWO)")
        print("\n")
        print(f"Original Text:\n{Text_Original}\n")
        print(f"Text:\n{Text_Final}\n")
        print("---------------------------------------------\n")
        Result_Predict = "SWO"

    return Text_Original , Text_Final, Result_Predict, Text_predict, Text_prob_predict, Percent_confident , Tokenize_Words





st.set_page_config(layout="wide")



st.title('SWO & Incident Prediction Identify by Pattara Model')

new_title = '<p style="font-family:sans-serif; color:Green; font-size: 42px;">Please fill information below, to Identify</p>'
st.markdown(new_title, unsafe_allow_html=True)


st.text('Please fill information below, to Identify')

TEXT = st.text_input("Enter your Text (required)")
Text_Prediction_Input = [TEXT]

if not TEXT:
  st.warning("Please fill out so required fields")



Text_Original , Text_Final, Result_Predict, Text_predict, Text_prob_predict, Percent_confident , Tokenize_Words = Text_Prediction_Completion(Text_Prediction = Text_Prediction_Input,  Tokenizer_Model = Load_Token, Prediction_Model = loaded_model)




if st.button('Check Text to Identification : SWO or Incident'):
    with st.spinner('Pattara Model will Identification the Result....'):
        time.sleep(4)
        

        if Text_predict == 1 :
            print(f"Prediction : {Text_predict}", "(Incident)")
            print("\n")
            print(f"Text:\n{Text_Final}\n")
            print("----------------------------------------------\n")
            
            st.header('Text to Identification is' )
            Result_Predict = '<p style="font-family:sans-serif; color:Red; font-size: 36px;">Incident Case</p>'
            st.markdown(Result_Predict,  unsafe_allow_html=True)



            st.write('### % Percent of Confident : ', str(Percent_confident), ' %')

            
            Data = [[str(Text_Original[0]),str(Text_Final[0]),str(Percent_confident),str(Text_prob_predict[0])]]

            df = pd.DataFrame(data = Data, columns = ['ข้อความ Original','ข้อความ After Clean','Percent_confident' ,'Prediction Margin ( 0 คือ SWO แต่ถ้าเป็น 1 คือ Incident)' ], index=["รายละเอียดของ Prediction"])
            pd.set_option('display.max_colwidth', None)

            st.table(df.head())

                      
            
        else :
            print(f"Prediction : {Text_predict}", "(SWO)")
            print("\n")
            print(f"Text:\n{Text_Final}\n")
            print("---------------------------------------------\n")
            Result_Predict = "SWO"
            st.header('Text to Identification is' )
            Result_Predict = '<p style="font-family:sans-serif; color:Yellow; font-size: 36px;">SWO Case</p>'
            st.markdown(Result_Predict,  unsafe_allow_html=True)

            st.write('### % Percent of Confident : ', str(Percent_confident), ' %')

            Data = [[str(Text_Original[0]),str(Text_Final[0]),str(Percent_confident),str(Text_prob_predict[0])]]

            df = pd.DataFrame(data = Data, columns = ['ข้อความ Original','ข้อความ After Clean','Percent_confident' ,'Prediction Margin ( 0 คือ SWO แต่ถ้าเป็น 1 คือ Incident)' ], index=["รายละเอียดของ Prediction"])
            pd.set_option('display.max_colwidth', None)

            st.table(df.head())
       




