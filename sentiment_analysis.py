#Input the relevant libraries
import streamlit as st
import altair as alt
import nltk
import numpy as np
import pandas as pd
import spacy
import random
from textblob import TextBlob
from nltk.tokenize.toktok import ToktokTokenizer
import re
from nltk.classify import accuracy as nltk_accuracy
from sklearn.utils import shuffle
import string
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from PIL import Image
import urllib.request

# Define the Streamlit app
def app():
    
    #do various initialization tasks
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    stopwords_list = stopwords.words('english')
    #no and not are excluded from stopwords
    stopwords_list.remove('no')
    stopwords_list.remove('not')
    tokenizer = ToktokTokenizer()

    st.title("Sentiment Analysis")      
    st.subheader("(c) 2023 Jimuel S. Servandil, BSCS 3A - AI")
    
    st.subheader('The TextBlob Package')
    st.write('TextBlob is a Python package for natural language processing (NLP) \
    tasks such as part-of-speech tagging, sentiment analysis, and text classification. \
    It is built on top of the popular Natural Language Toolkit (NLTK) and provides a \
    simple and intuitive API for performing various NLP tasks.')
    
    st.subheader('Sentiment Analysis')
    st.write("Sentiment analysis is the process of determining the emotional tone of a \
    piece of text. TextBlob provides two properties for sentiment analysis: polarity and \
    subjectivity. \n\nPolarity refers to the degree to which the text expresses a positive \
    or negative sentiment. Polarity is represented as a float value between -1.0 and 1.0, \
    where -1.0 represents a completely negative sentiment, 0.0 represents a neutral \
    sentiment, and 1.0 represents a completely positive sentiment./n/nSubjectivity, on the \
    other hand, refers to the degree to which the text expresses a subjective or objective \
    viewpoint. \n\nSubjectivity is also represented as a float value between 0.0 and 1.0, where \
    0.0 represents a completely objective viewpoint and 1.0 represents a completely \
    subjective viewpoint.")
    
    st.subheader('Balikatan 2023 Opinions Dataset')
    st.write('We load an opinion dataset in regards to the decision of the US and the Philippines \
    to launch the largest balikatan exercise, even though there is a rising tension in the region.\
    It contains two columns, first is the text which contains the content of the review, \
    and label - contains the 0 for negative and 1 for positive reviews. The \
    dataset contains 50 rows of data. We load the first 20 rows for viewing.')
    
    # with st.echo(code_location='below'):
        
    #The following function definitions show the codes need to perform each task
    def custom_remove_stopwords(text, is_lower_case=False):
        tokens = tokenizer.tokenize(text)
        tokens = [token.strip() for token in tokens]
        if is_lower_case:
            filtered_tokens = [token for token in tokens if token not in stopwords_list]
        else:
            filtered_tokens = [token for token in tokens if token.lower() not in stopwords_list]
        filtered_text = ' '.join(filtered_tokens)
        return filtered_text

    def remove_special_characters(text):
        text = re.sub('[^a-zA-z0-9\s]', '', text)
        return text

    def remove_html(text):
        import re
        html_pattern = re.compile('<.*?>')
        return html_pattern.sub(r' ', text)

    def remove_URL(text):
        url = re.compile(r'https?://\S+|www\.\S+')
        return url.sub(r' ', text)

    def remove_numbers(text):
        text =''.join([i for i in text if not i.isdigit()])
        return text

    nlp = []
    if 'en_core_web_sm' in spacy.util.get_installed_models():
        #disable named entity recognizer to reduce memory usage
        nlp = spacy.load('en_core_web_sm', disable=['ner'])
    else:
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load('en_core_web_sm', disable=['ner'])
    
    def remove_punctuations(text):
        for punctuation in string.punctuation:
            text = text.replace(punctuation, '')
        return text

    def cleanse(word):
        rx = re.compile(r'\D*\d')
        if rx.match(word):
            return ''      
        return word

    def remove_alphanumeric(strings):
        nstrings= [" ".join(filter(None, (cleanse(word) for word in string.split()))) \
                    for string in strings.split()]
        str1=' '.join(nstrings)
        return str1
    
    def lemmatize_text(text):
        text = nlp(text)
        text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
        return text
    
    if st.button('Load Dataset'):  
        df = pd.read_csv('balikatan2.csv')
        st.write(df.head(20))
        st.write('Dataset shape: ')
        st.text(df.shape)
        
        #Randomly select samples
        label_0=df[df['label']==0].sample(n=20)
        label_1=df[df['label']==1].sample(n=20)
        
        train = pd.concat([label_1, label_0])

        #remember this very useful function to randomly rearrange the dataset
        train = shuffle(train)
        
        st.write('We then randomly select 500 samples of positive reviews and \
        500 samples of negative reviews.  Remember the labels were added by \
        human reviewers and may not be the real sentiment.  We will use AI \
        to generate a new setiment value based on the analysis of the actual \
        text of the review.')
        
        st.write('We display the first 50 rows of the training dataset')
        st.write(train.head(50))
        
        st.write('Checking for null values. Do not proceed if we find a null value.')
        st.write(train.isnull().sum())
        
        st.write('We begin pre-processing the data.  The steps are necessary to clean up \
        the dataset and achieve better results from the classifier. Some steps are \
        resource-heavy so be patient and check the animated "running" indicator \
        at the upper right is showing that the page is still alive and running.')
        
        
        st.markdown("<div style='text-align:center'><img src='https://cdn.pixabay.com/animation/2022/12/05/15/23/15-23-06-837_512.gif'></div>", unsafe_allow_html=True)
        
        st.text('Doing pre-processing tasks...')

        st.text('Removing symbols...')
        train.replace(r'^\s*$', np.nan, regex=True, inplace=True)
        train.dropna(axis=0, how='any', inplace=True)

        st.text('Removing escape sequences...')
        train.replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"], value=["",""], regex=True, inplace=True)
        
        st.text('Removing non ascii data...')
        train['text']=train['text'].str.encode('ascii', 'ignore').str.decode('ascii')
        
        st.text('Removing punctuations...')
        train['text']=train['text'].apply(remove_punctuations)

        st.write('''
            In Natural Language Processing (NLP), stopwords refer to commonly occurring words in a language 
            that are often filtered out from the text before processing. These words typically do not 
            contribute much to the meaning of a sentence and are used primarily to connect other words 
            together. 
            
            Examples of stopwords in the English language include "the," "a," "an," "and," "in," "on," 
            "at," "for," "to," "of," "with," and so on.
        ''')

        
        st.text('Removing stop words...')
        train['text']=train['text'].apply(custom_remove_stopwords)
        
        st.text('Removing special characters...')
        train['text']=train['text'].apply(remove_special_characters)
        
        st.text('Removing HTML...')
        train['text']=train['text'].apply(remove_html)
        
        st.text('Removing URL...')
        train['text']=train['text'].apply(remove_URL)    
        
        st.text('Removing numbers...')
        train['text']=train['text'].apply(remove_numbers) 

        st.markdown("<div style='text-align:center'><img src='https://www.icegif.com/wp-content/uploads/icegif-2713.gif'></div>", unsafe_allow_html=True)

        
        st.text('\nWe look at our dataset after more pre-processing steps')
        st.write(train.head(50))    

        st.write('Removing alpha numeric data...')
        train['text']=train['text'].apply(remove_alphanumeric)
        st.text('We look at our dataset after the pre-processing steps')
        st.write(train.head(50))

        st.write('We lemmatize the words. \
                    \nThis process could take up to several minutes to complete. Please wait....')

        train['text']=train['text'].apply(lemmatize_text)
        
        #We use the TextBlob tweet sentiment function to get the sentiment
        train['sentiment']=train['text'].apply(lambda tweet: TextBlob(tweet).sentiment)
        
        st.write('We look at our dataset after more pre-processing steps')
        st.write(train.head(50))

        sentiment_series=train['sentiment'].tolist()
        columns = ['polarity', 'subjectivity']
        df1 = pd.DataFrame(sentiment_series, columns=columns, index=train.index)
        result = pd.concat([train, df1], axis=1)
        result.drop(['sentiment'],axis=1, inplace=True)

        result.loc[result['polarity']>=0.3, 'Sentiment'] = "Positive"
        result.loc[result['polarity']<0.3, 'Sentiment'] = "Negative"

        result.loc[result['label']==1, 'Sentiment_label'] = 1
        result.loc[result['label']==0, 'Sentiment_label'] = 0
        result.drop(['label'],axis=1, inplace=True)
        
        st.write('We view the dataset after the sentiment labels are updated.')
        result = result.sort_values(by=['Sentiment'], ascending=False)
        st.write(result.head(500))

        counts = result['Sentiment'].value_counts()
        st.write(counts)
        
        #reads the sample count from the previous line
        labels = ['Negative','Positive']
        sizes = [counts[0], counts[1]]
        custom_colours = ['#ff7675', '#74b9ff']

        fig = plt.figure(figsize=(8, 3), dpi=100)
        plt.subplot(1, 2, 1)
        plt.pie(sizes, labels = labels, textprops={'fontsize': 10}, startangle=140, \
                autopct='%1.0f%%', colors=custom_colours, explode=[0, 0.05])
        plt.subplot(1, 2, 2)
        sns.barplot(x = result['Sentiment'].unique(), y = result['Sentiment'].value_counts(), \
                palette= 'viridis')
        st.pyplot(fig)
        
        # Save the dataframe to a CSV file
        csv = result.to_csv(index=False)
        if csv:
            b64 = base64.b64encode(csv.encode()).decode()  # Convert to base64
            href = f'<a href="data:file/csv;base64,{b64}" download="data.csv">Download CSV file</a>'
            st.markdown(href, unsafe_allow_html=True)
        else:
            st.write("Error: Unable to generate CSV file.")

# run the app
if __name__ == "__main__":
    app()