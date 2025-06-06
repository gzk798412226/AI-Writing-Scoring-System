import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from data_processor import DataProcessor
from model import WritingScoringModel
from train import WritingDataset

def evaluate_model(model, test_loader, device):
    """
    评估模型性能
    Args:
        model: 模型实例
        test_loader: 测试数据加载器
        device: 设备
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            numerical_features = batch['numerical_features'].to(device)
            scores = batch['scores'].to(device)
            
            outputs = model(input_ids, attention_mask, numerical_features)
            
            all_predictions.extend(outputs.cpu().numpy())
            all_targets.extend(scores.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # 计算评估指标
    metrics = {}
    score_types = ['content', 'organization', 'expression', 'total']
    
    for i, score_type in enumerate(score_types):
        predictions = all_predictions[:, i]
        targets = all_targets[:, i]
        
        metrics[score_type] = {
            'mse': mean_squared_error(targets, predictions),
            'rmse': np.sqrt(mean_squared_error(targets, predictions)),
            'mae': mean_absolute_error(targets, predictions),
            'r2': r2_score(targets, predictions)
        }
    
    return metrics, all_predictions, all_targets

def plot_results(predictions, targets, score_types):
    """
    绘制预测结果与真实值的对比图
    Args:
        predictions: 预测值
        targets: 真实值
        score_types: 评分类型列表
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, score_type in enumerate(score_types):
        ax = axes[i]
        sns.scatterplot(x=targets[:, i], y=predictions[:, i], ax=ax)
        ax.plot([0, 1], [0, 1], 'r--', transform=ax.transAxes)
        ax.set_xlabel('True Scores')
        ax.set_ylabel('Predicted Scores')
        ax.set_title(f'{score_type.capitalize()} Score')
    
    plt.tight_layout()
    plt.savefig('evaluation_results.png')
    plt.close()

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 加载数据
    data_processor = DataProcessor('../dataset/evaluation')
    data_processor.load_data()
    X_train, X_test, y_train, y_test = data_processor.get_train_test_split()
    
    # 准备测试数据
    test_texts = X_test['text'].values
    test_numerical = X_test[['length', 'paragraph_num', 'sentence_num']].values
    test_scores = y_test.values
    
    # 加载模型
    model = WritingScoringModel()
    model.load_state_dict(torch.load('best_model.pth'))
    model = model.to(device)
    
    # 创建测试数据集和数据加载器
    test_dataset = WritingDataset(test_texts, test_numerical, test_scores, model.tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # 评估模型
    metrics, predictions, targets = evaluate_model(model, test_loader, device)
    
    # 打印评估结果
    print("\nEvaluation Results:")
    print("-" * 50)
    for score_type, score_metrics in metrics.items():
        print(f"\n{score_type.capitalize()} Score:")
        for metric_name, value in score_metrics.items():
            print(f"{metric_name.upper()}: {value:.4f}")
    
    # 绘制结果
    plot_results(predictions, targets, ['content', 'organization', 'expression', 'total'])

if __name__ == '__main__':
    main() 