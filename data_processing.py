import os
import pandas as pd
import re
import string
import torch

# Function to load data from a given folder path
def load_data_from_path(folder_path):
    data = []
    for label in ['pos', 'neg']:
        full_path = os.path.join(folder_path, label)
        for file_name in os.listdir(full_path):
            with open(os.path.join(full_path, file_name), 'r', encoding='utf-8') as file:
                lines = file.readlines()
                sentence = ' '.join(lines)
                data.append({'sentence': sentence, 'label': int(label == 'pos')})
    return pd.DataFrame(data)

# Paths to the dataset folders
folder_paths = {
    'train': './data/data_train/train',
    'valid': './data/data_train/test',
    'test': './data/data_test/test'
}


# Preprocessing function for text data
def preprocess_text(text):
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove punctuations and digits
    text = text.translate(str.maketrans('', '', string.punctuation + string.digits))
    # Remove emojis
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\u200d"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u23cf"
                               u"\u1f9d0"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    # Normalize whitespace
    text = ' '.join(text.split())
    # Lowercasing
    text = text.lower()
    return text


