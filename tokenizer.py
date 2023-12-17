from data_processing import *
from torchtext.vocab import build_vocab_from_iterator
import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence

class TextClassificationDataset(Dataset):
    def __init__(self, dataframe, vocab, tokenizer):
        self.labels = dataframe['label'].values
        tokenized_texts = [tokenizer(text) for text in dataframe['sentence'].values]
        self.texts = [torch.tensor([vocab[token] for token in tokenized_text], dtype=torch.long) for tokenized_text in tokenized_texts]
        self.max_seq_length = max(len(tokens) for tokens in tokenized_texts)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

def build_vocab(data_iter, tokenizer, specials=['<unk>', '<pad>']):
    """Build vocabulary from an iterable dataset."""
    tokenized_iter = map(tokenizer, data_iter)
    vocab = build_vocab_from_iterator(tokenized_iter, specials=specials)
    vocab.set_default_index(vocab['<unk>'])
    return vocab

def collate_batch(batch):
    label_list, text_list = [], []
    for _text, _label in batch:
        label_list.append(_label)
        text_list.append(_text)
    # auto define the longest review and set it for the length of the sequence
    text_list = pad_sequence(text_list, batch_first=True, padding_value=1) # Padding value 1 for <pad>
    return text_list, torch.tensor(label_list, dtype=torch.long)


if __name__ == '__main__':
    # loading the datasets
    train_df = load_data_from_path(folder_paths['train'])
    valid_df = load_data_from_path(folder_paths['valid'])
    test_df = load_data_from_path(folder_paths['test'])

    # preprocess dataset
    train_df['sentence'] = train_df['sentence'].apply(preprocess_text)
    valid_df['sentence'] = valid_df['sentence'].apply(preprocess_text)
    test_df['sentence'] = test_df['sentence'].apply(preprocess_text)

    # get basic tokenizer and build vocabulary from train_df
    tokenizer = get_tokenizer("basic_english")
    vocab = build_vocab(train_df['sentence'], tokenizer)

    # tokenize dataset
    train_dataset = TextClassificationDataset(train_df, vocab, tokenizer)
    valid_dataset = TextClassificationDataset(valid_df, vocab, tokenizer)
    test_dataset = TextClassificationDataset(test_df, vocab, tokenizer)

    # split into batches
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_batch)