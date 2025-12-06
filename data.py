from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence
import tiktoken
from omegaconf import DictConfig

# Padding to make all sequences in the batch of equal length
def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    target_ids = [item['target_ids'] for item in batch]
    
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    target_ids = pad_sequence(target_ids, batch_first=True, padding_value=0)
    
    return {
        'input_ids': input_ids,
        'target_ids': target_ids
    }
    
'''
If yoou don't have good Financial Training data just uncomment this function 
and modify the TextDataset class remove file_path and pass data from this function
'''
'''
# Let's just Load data from HF

def get_mixed_financial_data(sample_size=1000):
    """Mix of financial and general data for better generalization"""
    sources = []
    # 1. Financial specific
    try:
        fin_news = load_dataset("zeroshot/twitter-financial-news-tweets", 
                               split=f"train[:{sample_size//2}]")
        sources.extend(fin_news['text'])
    except:
        pass
    
    # 2. General knowledge base (for reasoning)
    wiki = load_dataset("wikitext", "wikitext-103-raw-v1", 
                       split=f"train[:{sample_size//2}]")
    sources.extend(wiki['text'])
    
    return sources

data = get_mixed_financial_data()
'''

class TextDataset(Dataset):
    def __init__(self,cfg:DictConfig, file_path, max_length, tokenizer_name):
        super().__init__()
        self.tokenizor_name = cfg.data.tokenizor_name
        self.tokenizer = tiktoken.get_encoding(self.tokenizer_name)
        self.max_length = max_length
        
        with open(file_path, 'r', encoding='utf-8') as f:
            self.full_text = f.read()
        
        self.all_tokens = self.tokenizer.encode(self.full_text)
    
    def __len__(self):
        return len(self.all_tokens) - self.max_length # We are creating a sliding window that's why we subtract max_length
    
    def __getitem__(self, idx):
        chunk = self.all_tokens[idx:idx + self.max_length + 1] # Adding one since target requires input shifted by one
        
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        target_ids = torch.tensor(chunk[1:], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'target_ids': target_ids
        }

def get_data_loader(cfg : DictConfig,file_path, batch_size, block_size, tokenizer_name):
    file_path = cfg.data.data_path
    batch_size = cfg.training.batch_size
    block_size = cfg.data.block_size
    tokenizor_name = cfg.data.tokenizer_name

    dataset = TextDataset(
        file_path=file_path,
        max_length=block_size,
        tokenizer_name=tokenizer_name
    )
    
    vocab_size = dataset.tokenizer.max_token_value + 1
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=4,
        pin_memory=True
    )
    
    return dataloader, vocab_size, dataset
