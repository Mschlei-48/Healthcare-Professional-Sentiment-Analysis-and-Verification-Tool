import pandas as pd
import requests
import streamlit as st
import regex as re
import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
import plotly.express as px
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
import plotly.graph_objects as go
from nltk.probability import FreqDist
from wordcloud import WordCloud
import matplotlib.pyplot as plt


st.set_page_config(
    layout="wide",  # Set the layout to wide
)
background_color = "#0000FF"  # Replace with your desired color code
text_color = "#333333"  # Replace with your desired text color

# Define the CSS style
custom_css = f"""
    <style>
        body {{
            background-color: {background_color};
        }}
    </style>
"""


# Inject the custom CSS into the Streamlit app
st.markdown(custom_css, unsafe_allow_html=True)
#Create a list that will store the data we will scrape
st.title("Twitter Scraping Web App")
st.sidebar.header("Get Tweets About A Certain Medical Professional")
st.sidebar.text("With this app you can get tweets \nabout any medical profeesional.The main aim \nof this app is to help verify \nmedical professionals. With this app you can \nsee what people are saying about a \ncertain medical professional. With this information and \nother pieces of information provided by Alchemy \nHealth you can make the best decision.")
#st.sidebar()
mat_data=[]
with st.sidebar.container():
    st.sidebar.markdown("**Search For a medical Professional**")
    query=st.sidebar.text_input(label="Give name of medical professional to query",value="Dr Matthew fake doctor")
    num=st.sidebar.slider(label="Number of tweets",min_value=10,max_value=100,value=100,step=20,key="num")
if query and num:
    payload = { 'api_key': 'f1db5231bec21e7705b5ba1fbf209d72', 
    'url': 'https://x.com/search?q=dr%20matthew%20bogus%20doctor&src=typed_query&f=top' }
    response= requests.get('https://api.scraperapi.com/', params=payload)
    data=response.json()

    #Extract the tweets
    for dat in data['organic_results']:
        mat_data.append(dat['snippet'])

    #Create a dataframe for the tweets
    df=pd.DataFrame(mat_data,columns=["Tweets"])


    for i in range(len(df)):
        df.loc[i,"Tweets"]=re.sub(r'[^\w\s]', '', df.loc[i,"Tweets"])
    for i in range(len(df)):
        df.loc[i,"Tweets"] = df.loc[i,"Tweets"].lower()
    def remove_stopwords(text):
        # Tokenize the text
        tokens = nltk.word_tokenize(text)

        # Remove stopwords
        filtered_tokens = [word for word in tokens if word.lower() not in stopwords.words('english')]

        # Join the filtered tokens back into a single string
        return ' '.join(filtered_tokens)
    df.loc["Tweets"] = df.loc[:,"Tweets"].apply(remove_stopwords)
    df.dropna(inplace=True)
    for i in range(len(df)):
        df.loc[i,"Tweets"] = re.sub(r'[^\x00-\x7F]+', '', df.loc[i,"Tweets"])
    for i in range(len(df)):
        df.loc[i,"Tweets"] = re.sub(r'\d+', '', df.loc[i,"Tweets"])


tab1, tab2=st.tabs(["Sentiments","Prediction"])
with tab1:
    col1,col2,col3=st.columns([0.4,0.4,0.2])
    with col3:
        st.markdown("**Choose Output**")
        button=st.radio("**Choose Output**",options=["Sentiments","Most Appearing Words"],index=0)
        if button=="Sentiments":
            with col1:
                def analyze_sentiment(text):
                    sia = SentimentIntensityAnalyzer()
                    sentiment_score = sia.polarity_scores(text)['compound']

                    if sentiment_score >= 0.05:
                        return 'Positive'
                    elif sentiment_score <= -0.05:
                        return 'Negative'
                    else:
                        return 'Neutral'

                # Apply sentiment analysis to the 'Tweets' column
                df['Sentiment'] = df['Tweets'].apply(analyze_sentiment)

                sentiment_counts = df['Sentiment'].value_counts()

                # Define colors for each sentiment
                colors = {'Positive': 'lime', 'Negative': 'red', 'Neutral': 'orange'}

                # Create a Plotly Pie chart
                fig = go.Figure()

                fig.add_trace(go.Pie(
                    labels=sentiment_counts.index,
                    values=sentiment_counts.values,
                    hole=0.6,  # Set the size of the hole for a doughnut plot
                    marker=dict(colors=[colors[sentiment] for sentiment in sentiment_counts.index]),
                ))

                fig.update_layout(
                    annotations=[dict(text='Sentiment', x=0.5, y=0.5, font_size=20, showarrow=False)],
                    width=250
                )

                # Display the Plotly chart in the Streamlit app
                st.markdown("**Sentiment Analysis Chart**")
                st.plotly_chart(fig)
            with col2:
                st.markdown("**Extracted Data**")
                st.dataframe(df)
        else:
            with col1:
                stop_words = set(stopwords.words('english'))

                def tokenize_and_remove_stopwords(text):
                    tokens = nltk.word_tokenize(text)
                    tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]
                    return tokens

                df['Tokens'] = df['Tweets'].apply(tokenize_and_remove_stopwords)

                # Flatten the list of tokens
                all_tokens = [token for sublist in df['Tokens'] for token in sublist]

                # Calculate word frequencies
                freq_dist = FreqDist(all_tokens)

                # Select the top N most frequent words
                top_words = freq_dist.most_common(10)  # Change 10 to the desired number of top words

                # Create a DataFrame for plotting
                word_df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])

                # Plot the vertical bar graph
                fig = px.bar(word_df, x='Word', y='Frequency', color='Word', color_discrete_map={'Word': 'lime'})
                fig.update_layout(title='Top Words Frequency', xaxis_title='Word', yaxis_title='Frequency',width=350)

                # Display the Plotly chart in the Streamlit app
                st.plotly_chart(fig)
            with col2:
                all_text = ' '.join(df['Tweets'])

                # Generate the word cloud with a black background and round shape
                wordcloud = WordCloud(
                    background_color='black',
                    contour_color='white',  # Set the contour color to white for a round shape
                    contour_width=1,         # Adjust the contour width
                    width=800,
                    height=400
                ).generate(all_text)

                # Display the word cloud using matplotlib
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.show()

                # Display the word cloud using Plotly
                fig = px.imshow(wordcloud.to_array(), binary_string=True, width=300, height=400)
                fig.update_layout(title='Word Cloud')
                st.plotly_chart(fig)

