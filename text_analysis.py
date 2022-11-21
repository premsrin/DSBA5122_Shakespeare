from PIL import Image
import numpy as np
import pandas as pd
import streamlit as st
import nltk
import altair as alt
nltk.download('punkt')

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from io import StringIO 
from nltk import word_tokenize
from nltk.probability import FreqDist

st.write('## Analyzing Shakespeare texts')

# Creating a dictionary not a list 
books = {" ":" ","A Mid Summer Night's Dream":"data/summer.txt","The Merchant of Venice":"data/merchant.txt","Romeo and Juliet":"data/romeo.txt"}
image = st.selectbox('Choose a txt file', books.keys())

image = books.get(image)

st.sidebar.header("Word Cloud Settings")
max_word = st.sidebar.slider("Max words", 10, 200,100, 10)
max_font = st.sidebar.slider("Size of Largest word", 50, 350,60)
image_size = st.sidebar.slider("image Width", 100, 800,400,10)
random = st.sidebar.slider("Random State", 30, 100,42)
stop_button = st.sidebar.checkbox("Remove Stop Words?")
word_count = st.sidebar.slider("Word Count Settings", 5, 100, 40, 5)

if image != " ":
    #Remove punctuations from raw text, many ways to do this
    dataset = open(image,"r").read().lower()

    stopwords=[]
    if stop_button: 
        stopwords = set(STOPWORDS)
        stopwords.update(['us', 'one', 'will', 'said', 'now', 'well', 'man', 'may',
        'little', 'say', 'must', 'way', 'long', 'yet', 'mean', 'thee' , 
        'put', 'seem', 'asked', 'made', 'half', 'much', 'o', 
        'certainly', 'might', 'came'])

    #Tokenize the dataset
    tokens = word_tokenize(dataset, language='english')
    
    # get the list without stop words
    words_ne=[]
    for word in tokens:
        if word not in stopwords and word.isalpha():
            words_ne.append(word)

    frequency = nltk.FreqDist(words_ne)
 
tab1, tab2, tab3 = st.tabs(['Word Cloud','Bar Chart', 'View Text'])

with tab1:
    if image != " ":
        cloud = WordCloud(background_color = "white", 
                            max_words = max_word, 
                            max_font_size=max_font, 
                            stopwords = stopwords, 
                            random_state=random)
        wc = cloud.generate(dataset)
        word_cloud = cloud.to_file('wordcloud.png')
        st.image(wc.to_array(), width = image_size)
    
with tab2:
    if image != " ":
        chart_data = pd.DataFrame(frequency.items(),columns=['word','count'])

        chart_data = chart_data[chart_data['count'] >= word_count]
        st.write(alt.Chart(chart_data,title='Bar Chart - Word Frequency').mark_bar().encode(
            x=alt.X('count:Q'),
            y=alt.Y('word:N',sort='-x'),
            tooltip=['count']
        ).interactive().properties(width=900))

with tab3:
    if image != " ":
        st.write(dataset)
