import json
import os
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

class DataProcessor:
    def __init__(self, data_dir: str):
        """
        初始化数据处理器
        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = data_dir
        self.data = []
        self.processed_data = None

    def load_data(self) -> None:
        """加载所有JSON文件数据"""
        print(f"Loading data from directory: {self.data_dir}")
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(self.data_dir, filename)
                print(f"Processing file: {filename}")
                try:
                    with open(file_path, 'r', encoding='utf-8-sig') as f:
                        data = json.load(f)
                        self.data.append(data)
                except Exception as e:
                    print(f"Error loading file {filename}: {str(e)}")
                    continue
        print(f"Successfully loaded {len(self.data)} files")

    def extract_features(self, document: Dict) -> Dict:
        """
        从文档中提取特征
        Args:
            document: 文档数据
        Returns:
            提取的特征字典
        """
        try:
            # 提取文本内容
            text = ""
            for paragraph in document['paragraph']:
                text += paragraph['form'] + " "

            # 提取元数据特征
            metadata = document['metadata']
            written_stat = metadata['written_stat']
            
            # 提取评分数据
            evaluation = document['evaluation']['evaluation_data']
            
            # 打印评分数据的结构
            print("Evaluation data structure:", json.dumps(evaluation, indent=2))
            
            # 计算各项分数
            content_score = 0
            organization_score = 0
            expression_score = 0
            total_score = 0
            
            # 内容分数
            if 'eva_score_con' in evaluation:
                con_scores = evaluation['eva_score_con']
                if 'evaluator1_score_total_con' in con_scores and 'evaluator2_score_total_con' in con_scores:
                    content_score = (con_scores['evaluator1_score_total_con'] + 
                                   con_scores['evaluator2_score_total_con']) / 2
            
            # 组织分数
            if 'eva_score_org' in evaluation:
                org_scores = evaluation['eva_score_org']
                if 'evaluator1_score_total_org' in org_scores and 'evaluator2_score_total_org' in org_scores:
                    organization_score = (org_scores['evaluator1_score_total_org'] + 
                                       org_scores['evaluator2_score_total_org']) / 2
            
            # 表达分数
            if 'eva_score_exp' in evaluation:
                exp_scores = evaluation['eva_score_exp']
                if 'evaluator1_score_total_exp' in exp_scores and 'evaluator2_score_total_exp' in exp_scores:
                    expression_score = (exp_scores['evaluator1_score_total_exp'] + 
                                     exp_scores['evaluator2_score_total_exp']) / 2
            
            # 总分
            if 'evaluator1_total_score' in evaluation and 'evaluator2_total_score' in evaluation:
                total_score = (evaluation['evaluator1_total_score'] + 
                             evaluation['evaluator2_total_score']) / 2
            
            features = {
                'text': text.strip(),
                'length': written_stat['written_length'],
                'paragraph_num': written_stat['paragraph_num'],
                'sentence_num': written_stat['sentence_num'],
                'content_score': content_score,
                'organization_score': organization_score,
                'expression_score': expression_score,
                'total_score': total_score
            }
            
            return features
            
        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            print("Document structure:", json.dumps(document, indent=2))
            raise

    def process_data(self) -> pd.DataFrame:
        """
        处理所有数据并转换为DataFrame格式
        Returns:
            处理后的DataFrame
        """
        processed_data = []
        
        for data in self.data:
            for document in data['document']:
                try:
                    features = self.extract_features(document)
                    processed_data.append(features)
                except Exception as e:
                    print(f"Error processing document: {str(e)}")
                    continue
        
        self.processed_data = pd.DataFrame(processed_data)
        return self.processed_data

    def get_train_test_split(self, test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        将数据分割为训练集和测试集
        Args:
            test_size: 测试集比例
            random_state: 随机种子
        Returns:
            训练集和测试集的元组
        """
        if self.processed_data is None:
            self.process_data()
            
        from sklearn.model_selection import train_test_split
        
        X = self.processed_data[['text', 'length', 'paragraph_num', 'sentence_num']]
        y = self.processed_data[['content_score', 'organization_score', 'expression_score', 'total_score']]
        
        return train_test_split(X, y, test_size=test_size, random_state=random_state) 