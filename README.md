# AI Writing Scoring System

这是一个基于深度学习的写作评分系统，可以自动评估文章的内容、组织和表达质量。

## 功能特点

- 多维度评分：内容、组织、表达三个维度
- 支持中文和韩语文本评估
- 基于 BERT 的文本特征提取
- 详细的评分报告

## 安装步骤

1. 克隆仓库：
```bash
git clone [your-repository-url]
cd [repository-name]
```

2. 创建虚拟环境：
```bash
conda create -n wy python=3.10
conda activate wy
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

1. 训练模型：
```bash
python train.py
```

2. 评估模型：
```bash
python evaluate.py
```

3. 预测单个文本：
```bash
python predict.py
```

## 项目结构

```
code/
├── data_processor.py    # 数据处理模块
├── model.py            # 模型定义
├── train.py           # 训练脚本
├── evaluate.py        # 评估脚本
├── predict.py         # 预测脚本
└── requirements.txt   # 依赖文件
```

## 评分标准

- 内容评分 (Content Score)：评估文章的主题、论点和论据
- 组织评分 (Organization Score)：评估文章的结构和逻辑性
- 表达评分 (Expression Score)：评估语言使用的准确性和流畅性

## 注意事项

- 确保已安装所有依赖包
- 训练模型需要 GPU 支持
- 预测时使用训练好的模型文件 (best_model.pth)

## License

MIT License 