from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence
import tiktoken

def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    target_ids = [item['target_ids'] for item in batch]
    
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    target_ids = pad_sequence(target_ids, batch_first=True, padding_value=0)
    
    return {
        'input_ids': input_ids,
        'target_ids': target_ids
    }

class TextDataset(Dataset):
    def __init__(self, file_path, max_length, tokenizer_name='cl100k_base'):
        super().__init__()
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        self.max_length = max_length
        
        with open(file_path, 'r', encoding='utf-8') as f:
            self.full_text = f.read()
        
        self.all_tokens = self.tokenizer.encode(self.full_text)
    
    def __len__(self):
        return len(self.all_tokens) - self.max_length
    
    def __getitem__(self, idx):
        chunk = self.all_tokens[idx:idx + self.max_length + 1]
        
        input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
        target_ids = torch.tensor(chunk[1:], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'target_ids': target_ids
        }

def get_data_loader(file_path, batch_size=32, block_size=512, tokenizer_name='cl100k_base'):
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