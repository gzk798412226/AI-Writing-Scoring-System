import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
from data_processor import DataProcessor
from model import WritingScoringModel

class WritingDataset(Dataset):
    def __init__(self, texts, numerical_features, scores, tokenizer, max_length=512):
        self.texts = texts
        self.numerical_features = numerical_features
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        numerical = self.numerical_features[idx]
        score = self.scores[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'numerical_features': torch.tensor(numerical, dtype=torch.float32),
            'scores': torch.tensor(score, dtype=torch.float32)
        }

def train_model(model, train_loader, val_loader, device, num_epochs=10, learning_rate=2e-5):
    """
    训练模型
    Args:
        model: 模型实例
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        device: 训练设备
        num_epochs: 训练轮数
        learning_rate: 学习率
    """
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_steps = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} - Training'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            numerical_features = batch['numerical_features'].to(device)
            scores = batch['scores'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, numerical_features)
            loss = criterion(outputs, scores)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_steps += 1
        
        avg_train_loss = train_loss / train_steps
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} - Validation'):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                numerical_features = batch['numerical_features'].to(device)
                scores = batch['scores'].to(device)
                
                outputs = model(input_ids, attention_mask, numerical_features)
                loss = criterion(outputs, scores)
                
                val_loss += loss.item()
                val_steps += 1
        
        avg_val_loss = val_loss / val_steps
        
        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Average Training Loss: {avg_train_loss:.4f}')
        print(f'Average Validation Loss: {avg_val_loss:.4f}')
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print('Best model saved!')

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 加载数据
    data_processor = DataProcessor('../dataset/evaluation')
    data_processor.load_data()
    X_train, X_test, y_train, y_test = data_processor.get_train_test_split()
    
    # 准备数据
    train_texts = X_train['text'].values
    train_numerical = X_train[['length', 'paragraph_num', 'sentence_num']].values
    train_scores = y_train.values
    
    test_texts = X_test['text'].values
    test_numerical = X_test[['length', 'paragraph_num', 'sentence_num']].values
    test_scores = y_test.values
    
    # 初始化模型和tokenizer
    model = WritingScoringModel()
    tokenizer = model.tokenizer
    
    # 创建数据集和数据加载器
    train_dataset = WritingDataset(train_texts, train_numerical, train_scores, tokenizer)
    test_dataset = WritingDataset(test_texts, test_numerical, test_scores, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # 将模型移到设备
    model = model.to(device)
    
    # 训练模型
    train_model(model, train_loader, test_loader, device)

if __name__ == '__main__':
    main()
