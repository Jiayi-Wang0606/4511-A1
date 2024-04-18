# Setup
"""

pip install bertopic

pip install topic-wizard

!pip3 install torch==2.2.0

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots
from matplotlib.colors import LinearSegmentedColormap

import seaborn as sns
import torch

from bertopic import BERTopic

import os
from google.colab import drive
drive.mount('/content/drive')

path = "/content/drive/My Drive"

os.chdir(path)

"""## fourth model - 日期处理后（2022-01-01 - 2022-05-01）"""

kaggle_new = pd.read_csv('/content/drive/My Drive/Colab Notebooks/4511/kaggle_new.csv')
kaggle_new['timestamp'] = pd.to_datetime(kaggle_new['date'])

kaggle_new.head()

"""## fifth model"""

# Apply to all data
# Randomly sample 2000 news articles for each year from 2012 to 2018
# Keep the data unchanged for 2019 and later

# Extract data from 2012 to 2018
data_2012_to_2018 = kaggle_new[kaggle_new['year'] <= 2018]

# Create an empty DataFrame to store sampled data
sampled_data_2012_to_2018 = pd.DataFrame()

# Randomly sample 2000 articles for each year from 2012 to 2018
for year in range(2012, 2019):
    year_data = data_2012_to_2018[data_2012_to_2018['year'] == year]
    sampled_data_2012_to_2018 = pd.concat([sampled_data_2012_to_2018, year_data.sample(n=2000, random_state=42)], ignore_index=True)

# Extract data from 2019 onwards
data_2019_onwards = kaggle_new[kaggle_new['year'] > 2018]

# Merge sampled data and unchanged data
final_data = pd.concat([sampled_data_2012_to_2018, data_2019_onwards], ignore_index=True)

final_data.dropna(subset=['text'], inplace=True)

start_date = pd.Timestamp(2015, 1, 1)
end_date = pd.Timestamp(2016, 1, 1)
filtered_data = final_data[(final_data['timestamp'] >= start_date) & (final_data['timestamp'] < end_date)].reset_index(drop=True)

docs = filtered_data['text']
timestamps = filtered_data['month']

topic_model3 = BERTopic()
topic_labels3,probability =  topic_model3.fit_transform(docs)

topics_over_time3 = topic_model3.topics_over_time(docs, timestamps)
topic_model3.visualize_topics_over_time(topics_over_time3,top_n_topics=10)

topic_labels3
topic_model3._extract_embeddings(docs)

topic_model3.topics_

"""#Right Model"""

import sys
import topicwizard
sys.path.append('/content/drive/My Drive/Colab Notebooks/4511')
from topic_wizard_bertopic import BERTopicWrapper

model = BERTopic(language="english")
wrapped_model = BERTopicWrapper(model)
topic_data = wrapped_model.prepare_topic_data(docs)

from topicwizard.figures import topic_map
topic_map(topic_data)

topicwizard.visualize(docs, model=wrapped_model)

#Visualize on fitted model
wrapped_model3 = BERTopicWrapper(topic_model3)
topic_data3 = wrapped_model3.prepare_topic_data(docs)
topicwizard.visualize(topic_data=topic_data3)

topic_data3

loading_dir = '/content/drive/My Drive/Colab Notebooks/4511/'
loaded_model = BERTopic.load(loading_dir+"saved_model")

loaded_model.get_topic_info()

loaded_model.topics_

import pickle

# 打开 pickle 文件进行读取
with open(loading_dir + 'dataset.pickle', 'rb') as f:
    # 使用 pickle.load() 方法加载 pickle 文件中的数据
    dataset = pickle.load(f)

loading_dir='/content/drive/My Drive/Colab Notebooks/4511/'

# load data
class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.raw_data = pd.read_json(loading_dir+"News_Category_Dataset_v3.json", lines=True)

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i):
        text = self.raw_data['headline'].iloc[i] + ' | ' + self.raw_data['short_description'].iloc[i]
        label = self.raw_data['category'].iloc[i]
        timestamp = self.raw_data['date'].iloc[i]


        return text, label, timestamp


print("loading data...")
dataset = Dataset()
len(dataset)# 1. load data
class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.raw_data = pd.read_json(loading_dir+"News_Category_Dataset_v3.json", lines=True)

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i):
        text = self.raw_data['headline'].iloc[i] + ' | ' + self.raw_data['short_description'].iloc[i]
        label = self.raw_data['category'].iloc[i]
        timestamp = self.raw_data['date'].iloc[i]


        return text, label, timestamp


print("loading data...")
dataset = Dataset()
len(dataset)

documents=dataset[:][0]

wrapped_model4 = BERTopicWrapper(loaded_model)
topic_data4 = wrapped_model4.prepare_topic_data(documents)
topicwizard.visualize(topic_data=topic_data4)

topic_data4













"""# Wrong Version"""

import sys
import topicwizard
sys.path.append('/content/drive/My Drive/Colab Notebooks/4511')
from topic_wizard_bertopic import BERTopicWrapper

model = BERTopic(language="english")
wrapped_model = BERTopicWrapper(model)
topicwizard.visualize(docs, model=wrapped_model)

model = BERTopic(language="english")
wrapped_model = BERTopicWrapper(model)
topic_data = wrapped_model.prepare_topic_data(docs)

from topicwizard.figures import topic_map
topic_map(topic_data)

model.topics_

#Original import
'''from topicwizard.compatibility import BERTopicWrapper
model = BERTopic(language="english")
wrapped_model = BERTopicWrapper(model)
topic_data = wrapped_model.prepare_topic_data(docs)'''

if self.model.topic_labels_:
  topic_names = [self.model.topic_labels_[topic] for topic in self.model.topics_]
else:
  topic_names = self.model.generate_topic_labels(nr_words=3)

topic_data.keys()

document_term_matrix = topic_data['document_term_matrix']
document_topic_matrix = topic_data['document_topic_matrix']
topic_term_matrix = topic_data['topic_term_matrix']
topic_names = topic_data['topic_names']
transform = topic_data['transform']
document_representation = topic_data['document_representation']

document_representation

document_lengths = document_term_matrix.sum(axis=1)
# Calculating an estimate of empirical topic frequencies
topic_importances = (document_topic_matrix.T * document_lengths).sum(axis=1)
topic_importances = np.squeeze(np.asarray(topic_importances))
# Calculating empirical estimate of term-topic frequencies
topic_term_importances = (topic_term_matrix.T * topic_importances).T



