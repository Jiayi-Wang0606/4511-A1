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

import sys
import topicwizard
sys.path.append('/content/drive/My Drive/Colab Notebooks/4511')
from topic_wizard_bertopic import BERTopicWrapper # topic_wizard module modified by us
"""

# load model
loading_dir = '/content/drive/My Drive/Colab Notebooks/4511/'
loaded_model = BERTopic.load(loading_dir+"saved_model")

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
len(dataset)
documents=dataset[:][0]

# Prepare topic_data for visualization
wrapped_model = BERTopicWrapper(loaded_model)
topic_data = wrapped_model.prepare_topic_data(documents)

# Change topic_names to llama2 topic
topic_info = loaded_model.get_topic_info()
topic_info['Key_word'] = topic_info['Llama2'].apply(lambda x: x[0])
topic_data['topic_names'] = topic_info.loc[loaded_model.topic_sizes_.keys(),'Key_word'].tolist()

# Visualize the adjusted data with topic_wizard
topicwizard.visualize(topic_data=topic_data)


