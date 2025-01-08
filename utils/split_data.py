import pandas as pd
from sklearn.model_selection import train_test_split

def split_csv(file_path, train_file, test_file):
    # 读取CSV文件
    data = pd.read_csv(file_path)
    
    # 按7:3划分数据集
    train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)
    
    # 保存到新的CSV文件中
    train_data.to_csv(train_file, index=False)
    test_data.to_csv(test_file, index=False)

# 示例用法
file_path = "/home/ywc/projects/graph_cnn/data/kinase_final_data/Kinase_final_data.csv"  # 原始CSV文件路径
train_file = "/home/ywc/projects/graph_cnn/data/kinase_final_data/train.csv"  # 训练集保存路径
test_file = "/home/ywc/projects/graph_cnn/data/kinase_final_data/val.csv"  # 测试集保存路径

split_csv(file_path, train_file, test_file)
