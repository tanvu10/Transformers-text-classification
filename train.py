from model import *
from torch.utils.data import DataLoader
from data_processing import *
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from tokenizer import *
from trainer import *

class Config(object):
    batch_size = 64
    d_model = 200 # embedded_size
    nhead = 8
    num_encoder_layers = 2
    dim_feedforward = 128
    dropout = 0.1
    max_seq_length = None
    lr = 0.001
    vocab_size = None
    max_epochs = 3
    early_stopping = 2

config = Config()
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
config.vocab_size = len(vocab)
# max_seq_length_from_train = train_dataset.max_seq_length


# tokenize dataset
train_dataset = TextClassificationDataset(train_df, vocab, tokenizer)
valid_dataset = TextClassificationDataset(valid_df, vocab, tokenizer)
test_dataset = TextClassificationDataset(test_df, vocab, tokenizer)

# get max sequence length
max_seq_length = max(
    train_dataset.max_seq_length,
    valid_dataset.max_seq_length,
    test_dataset.max_seq_length
)
config.max_seq_length = max_seq_length


# hyper-parameters tuning
best_config = tune_hyperparameters(config, train_dataset, valid_dataset)

# combine train + valid
final_train_df = pd.concat([train_df, valid_df])
final_train_dataset = TextClassificationDataset(final_train_df, vocab, tokenizer)
final_train_loader = DataLoader(final_train_dataset, batch_size=best_config.batch_size, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(test_dataset, batch_size=best_config.batch_size, collate_fn=collate_batch)

# final train
save_model_path = './final_model'
os.makedirs(save_model_path, exist_ok=True)
final_model = TransformerEncoderModel(best_config)
optimizer = optim.Adam(final_model.parameters(), lr=best_config.lr)
criterion = CrossEntropyLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


train(final_model, optimizer, criterion, final_train_loader, test_loader, best_config.max_epochs, save_model_path, option='test')
test_loss, test_accuracy = evaluate(final_model, test_loader, device, criterion)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

