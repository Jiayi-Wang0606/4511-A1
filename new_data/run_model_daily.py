import pandas as pd
import numpy as np
import torch
import os
import pickle
import openai
import bertopic
import tiktoken
from datetime import datetime
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.representation import OpenAI
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from bertopic.vectorizers import OnlineCountVectorizer
from umap import UMAP
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from bertopic import BERTopic
import bisect
from tqdm import tqdm
import matplotlib.pyplot as plt



class Dataset(torch.utils.data.Dataset):
    def __init__(self,data_dir):
        self.raw_data = pd.read_csv(data_dir)
        self.raw_data['description'].fillna('', inplace=True)
        self.raw_data.dropna(subset=['title','publishedDate'], inplace=True)

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i):
        text = self.raw_data['title'].iloc[i] + ' | ' + self.raw_data['description'].iloc[i]
        timestamps = self.raw_data['publishedDate'].iloc[i].split('T')[0]
        tickers=self.raw_data['tickers'].iloc[i]

        return text, timestamps, tickers


def update_model_OlineBertopic(loading_dir='/content/drive/MyDrive/Topic Mining Project/new_data/'):  ## change to your path
    print("loading data...")
    dataset = Dataset(loading_dir+'news_data/news_with_ticker.csv')

    documents = [dataset[i][0] for i in range(len(dataset))]
    timestamp = [dataset[i][1] for i in range(len(dataset))]
    all_dates=sorted(np.unique(timestamp))
    doc_chunks = [documents[bisect.bisect_left(timestamp, date):bisect.bisect_right(timestamp, date)] for date in all_dates]
    date_chunks = [timestamp[bisect.bisect_left(timestamp, date):bisect.bisect_right(timestamp, date)] for date in all_dates]
    print('update model using data from: ',all_dates[-1])

    loaded_model = BERTopic.load(loading_dir+"model_data/updated_model")
    topics=loaded_model.topics_
    docs=documents[bisect.bisect_left(timestamp, all_dates[-1]):]
    if len(docs)>0:
        loaded_model.partial_fit(docs)
        topics.extend(loaded_model.topics_)
        loaded_model.topics_=topics

    current_date = datetime.now().strftime("%Y-%m-%d")
    loaded_model.save(loading_dir+"model_data/updated_model", serialization="pickle")
    loaded_model.save(loading_dir+"model_data/model_"+current_date, serialization="pickle")

    # topics_over_time = loaded_model.topics_over_time(documents, timestamp, nr_bins=20)
    # loaded_model.visualize_topics_over_time(topics_over_time, top_n_topics=10)



def update_model(loading_dir='/content/drive/MyDrive/Topic Mining Project/new_data/'):

    dataset = Dataset(loading_dir+'news_data/news_with_ticker.csv')
    documents = [dataset[i][0] for i in range(len(dataset))]
    timestamp = [dataset[i][1] for i in range(len(dataset))]
    all_dates=sorted(np.unique(timestamp))

    dates_in_files = [file[-10:] for file in os.listdir(loading_dir+"model_data/")]
    dates_in_files.sort()
    unfitted_dates=[date for date in all_dates if date not in dates_in_files]

    for date in unfitted_dates:
        print(date)
        vectorizer_model=CountVectorizer(stop_words="english")
        umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
        kmeans_model = KMeans(n_clusters=50)
        model=BERTopic(
                vectorizer_model=vectorizer_model,
                umap_model=umap_model,
                hdbscan_model=kmeans_model,
                verbose=False,
                )
        model.fit(documents[bisect.bisect_left(timestamp, date):bisect.bisect_right(timestamp, date)])
        model.save(loading_dir+"model_data/model_"+date, serialization="pickle")


def merge_analysis(loading_dir='/content/drive/MyDrive/Topic Mining Project/new_data/'):

    dataset = Dataset(loading_dir+'news_data/news_with_ticker.csv')
    documents = [dataset[i][0] for i in range(len(dataset))]
    timestamp = [dataset[i][1] for i in range(len(dataset))]
    all_dates=sorted(np.unique(timestamp))

    topic_models=[]
    for file in os.listdir(loading_dir+"model_data/"):
        topic_models.append(BERTopic.load(loading_dir+"model_data/"+file))
    merged_model = BERTopic.merge_models(topic_models,min_similarity=0.5)  

    ts_df=merged_model.get_document_info(documents).copy()
    ts_df['time']=timestamp

    cnt_df=pd.DataFrame()
    for i,date in enumerate(all_dates):
        df=ts_df[ts_df['time']==date]
        df=df.groupby('Name')['Document'].count().sort_values(ascending=False)[:20]
        df=df.reset_index()
        df.columns=['Name','Count']
        df['Date']=date
        cnt_df=pd.concat([cnt_df,df])

    plt.figure(figsize=(15, 20))
    cnt_df.pivot(index='Date', columns='Name', values='Count')[ts_df.groupby('Name')['Document'].count().sort_values(ascending=False)[:20].index].plot()
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.title('Counts over Time')
    plt.legend(loc='upper left',bbox_to_anchor=(1.05, 1))  
    plt.grid(True) 
    plt.show()


if __name__=='__main__':
    update_model()