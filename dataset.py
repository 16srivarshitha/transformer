import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

class TranslationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Access translation pairs correctly
        src_tokens = self.tokenizer.encode(
            item['translation']['en'], 
            max_length=self.max_length, 
            truncation=True, 
            padding='max_length'
        )
        
        tgt_tokens = self.tokenizer.encode(
            item['translation']['de'], 
            max_length=self.max_length, 
            truncation=True, 
            padding='max_length'
        )
        
        return {
            'src': torch.tensor(src_tokens, dtype=torch.long),
            'tgt': torch.tensor(tgt_tokens, dtype=torch.long)
        }

def create_dataloaders(model_config,training_config):
    # Load dataset
    dataset = load_dataset('opus100', 'de-en', split='train').select(range(10000))
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    
    # Split data
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_data = dataset.select(range(train_size))
    val_data = dataset.select(range(train_size, train_size + val_size))
    
    # Create datasets
    train_dataset = TranslationDataset(train_data, tokenizer, model_config.max_seq_len)
    val_dataset = TranslationDataset(val_data, tokenizer, model_config.max_seq_len)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size= training_config.batch_size, 
        shuffle=True, 
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size= training_config.batch_size, 
        shuffle=False, 
        num_workers=2
    )
    
    return train_loader, val_loader, tokenizer