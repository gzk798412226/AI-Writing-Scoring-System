import torch
from model import WritingScoringModel
from transformers import BertTokenizer
import numpy as np
import json

def predict_score(text: str, model_path: str = 'best_model.pth'):
    """
    预测单个文本的评分
    Args:
        text: 输入文本
        model_path: 模型路径
    Returns:
        预测的评分结果
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 加载模型和tokenizer
    model = WritingScoringModel()
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    
    # 准备输入数据
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # 计算文本特征
    length = len(text)
    paragraph_num = text.count('\n') + 1
    sentence_num = text.count('。') + text.count('！') + text.count('？') + 1
    
    numerical_features = torch.tensor([[length, paragraph_num, sentence_num]], dtype=torch.float32).to(device)
    
    # 预测
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, numerical_features)
        scores = outputs.cpu().numpy()[0]
    
    # 将 numpy float32 转换为 Python float
    content_score = float(scores[0])
    org_score = float(scores[1])
    exp_score = float(scores[2])
    total_score = float(scores[3])
    
    # 格式化输出结果
    result = {
        'eva_score_con': {
            'evaluator1_score_total_con': round(content_score, 2),
            'evaluator1_score_con1': round(content_score * 0.2, 2),
            'evaluator1_score_con2': round(content_score * 0.2, 2),
            'evaluator1_score_con3': round(content_score * 0.2, 2),
            'evaluator1_score_con4': round(content_score * 0.2, 2),
            'evaluator1_score_con5': round(content_score * 0.2, 2)
        },
        'eva_score_org': {
            'evaluator1_score_total_org': round(org_score, 2),
            'evaluator1_score_org1': round(org_score * 0.5, 2),
            'evaluator1_score_org2': round(org_score * 0.5, 2)
        },
        'eva_score_exp': {
            'evaluator1_score_total_exp': round(exp_score, 2),
            'evaluator1_score_exp1': round(exp_score * 0.5, 2),
            'evaluator1_score_exp2': round(exp_score * 0.5, 2)
        },
        'evaluator1_total_score': round(total_score, 2)
    }
    
    return result

def main():
    # 韩语示例文本
    sample_text = """
    인공지능의 발전과 미래
    
    인공지능 기술은 우리의 일상생활에 큰 변화를 가져왔습니다. 스마트폰, 자율주행차, 음성인식 등 다양한 분야에서 AI가 활용되고 있습니다.
    
    특히 의료 분야에서는 AI가 질병 진단과 치료에 큰 도움을 주고 있습니다. 대량의 의료 데이터를 분석하여 의사들이 놓칠 수 있는 미세한 증상을 발견할 수 있습니다. 교육 분야에서는 AI가 각 학생의 학습 특성에 맞는 맞춤형 학습 방안을 제공하고 있습니다.
    
    하지만 AI의 발전은 여러 도전 과제도 안고 있습니다. 첫째, 일자리 문제입니다. 많은 전통적인 직업이 AI로 대체될 수 있습니다. 둘째, 개인정보 보호 문제입니다. AI 시스템은 학습을 위해 많은 개인 데이터가 필요하며, 이는 데이터 보안에 대한 우려를 불러일으킵니다.
    
    결론적으로, AI 기술은 양날의 검입니다. 우리는 AI를 발전시키면서도 그로 인한 위험을 방지하는 데 주의해야 합니다. 그래야만 AI가 진정으로 인류에게 이로움을 줄 수 있을 것입니다.
    """
    
    # 预测评分
    scores = predict_score(sample_text)
    
    # 打印结果
    print("\n评分结果:")
    print("-" * 30)
    print(json.dumps(scores, indent=2, ensure_ascii=False))

if __name__ == '__main__':
    main() 