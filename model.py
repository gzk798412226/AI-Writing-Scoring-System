import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import numpy as np
from typing import Dict, List, Tuple

class WritingScoringModel(nn.Module):
    def __init__(self, bert_model_name: str = 'bert-base-multilingual-cased', dropout: float = 0.1):
        """
        初始化写作评分模型
        Args:
            bert_model_name: BERT模型名称
            dropout: Dropout比率
        """
        super(WritingScoringModel, self).__init__()
        
        # BERT模型用于文本特征提取
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        
        # 特征维度
        self.bert_dim = self.bert.config.hidden_size
        self.numerical_dim = 3  # length, paragraph_num, sentence_num
        
        # 评分预测层
        self.scoring_layers = nn.Sequential(
            nn.Linear(self.bert_dim + self.numerical_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 4)  # 4个评分维度：内容、组织、表达、总分
        )
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                numerical_features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            input_ids: 输入文本的token ids
            attention_mask: 注意力掩码
            numerical_features: 数值特征
        Returns:
            预测的评分
        """
        # 获取BERT输出
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text_features = bert_output.last_hidden_state[:, 0, :]  # 使用[CLS]标记的输出
        
        # 合并文本特征和数值特征
        combined_features = torch.cat([text_features, numerical_features], dim=1)
        
        # 预测评分
        scores = self.scoring_layers(combined_features)
        return scores
    
    def predict(self, text: str, length: int, paragraph_num: int, sentence_num: int) -> Dict[str, float]:
        """
        预测单篇文章的评分
        Args:
            text: 文章文本
            length: 文章长度
            paragraph_num: 段落数
            sentence_num: 句子数
        Returns:
            预测的评分字典
        """
        self.eval()
        with torch.no_grad():
            # 处理文本输入
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            
            # 处理数值特征
            numerical_features = torch.tensor([[length, paragraph_num, sentence_num]], dtype=torch.float32)
            
            # 预测评分
            scores = self(input_ids, attention_mask, numerical_features)
            scores = scores.squeeze().numpy()
            
            return {
                'content_score': float(scores[0]),
                'organization_score': float(scores[1]),
                'expression_score': float(scores[2]),
                'total_score': float(scores[3])
            } 