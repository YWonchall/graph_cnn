import numpy as np
import pandas as pd
import argparse
import torch
from torch_geometric.data import DataLoader
import copy
import os

from utils.smiles_to_graph import make_regre_mol
from utils.smiles_to_graph import make_regre_vec
from utils.viz_utils import plot_loss_curve, plot_metric_curve, plot_result_distribution
from runner import experiment
import model
from utils.utils import load_config
from resnet import ResGraphNet

# warnings.filterwarnings(action='ignore')
parser = argparse.ArgumentParser(
    description="Graph Convolutional Network for logS Regression", 
    epilog="python regression.py -D './myDataset.xlsx' -X1 'Solute SMILES'  -Y 'LogS' -O './results/myResult.json' -M './results/myModel.pt'")
parser.add_argument('--config', '-c', type=str, help='config path')
args = parser.parse_args()

configs = load_config(args.config)

os.makedirs(configs.output_folder, exist_ok=True)

np.random.seed(configs.seed)
torch.manual_seed(configs.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


print()
print('Graph Convolutional Network for logS Regression')
print('Soongsil University, Seoul, South Korea')
print('Computational Science and Artificial Intelligence Lab')
print()
print('[Preparing Data]')
print('- Device :', device)
print()


df_train = pd.read_csv(configs.train_path)
df_train = pd.concat([df_train[configs.input_name], df_train[configs.label_name]], axis=1)
df_train.columns = ['Solute SMILES',  'logS']
df_train = df_train.dropna(axis=0).reset_index(drop=True)

df_val = pd.read_csv(configs.val_path)
df_val = pd.concat([df_val[configs.input_name], df_val[configs.label_name]], axis=1)
df_val.columns = ['Solute SMILES',  'logS']
df_val = df_val.dropna(axis=0).reset_index(drop=True)


print('[Converting to Graph]')
train_mols1_key, train_mols_value = make_regre_mol(df_train)
test_mols1_key, test_mols_value = make_regre_mol(df_val)

train_X1 = make_regre_vec(train_mols1_key, train_mols_value)
test_X1 = make_regre_vec(test_mols1_key, test_mols_value)
print(test_X1)

train_X = []
for i in range(len(train_X1)):
    train_X.append([train_X1[i]])
test_X = []
for i in range(len(test_X1)):
    test_X.append([test_X1[i]])

print('- Train Data :', len(train_X))
print('- Test Data :', len(test_X))

#训练集和测试集的数据加载器
train_loader = DataLoader(train_X, batch_size=configs.batch_size, shuffle=True, drop_last=False)
test_loader = DataLoader(test_X, batch_size=len(test_X), shuffle=True, drop_last=False)

model = model.Net(configs)
# model = ResGraphNet(res_layers=[2, 2, 2, 2])
model = model.to(device)

print()
dict_result = dict()
result = experiment(model, train_loader, test_loader, device, configs)

save_prefix = os.path.join(configs.output_folder, configs.exp_name)
dict_result[configs.exp_name] = copy.deepcopy(result)
result_df = pd.DataFrame(dict_result).transpose()
result_df.to_json(save_prefix + ".json", orient='table')

metric_values = dict(
    mae=result_df['list_mae'].iloc[0],
    mse=result_df['list_mse'].iloc[0],
    r_square=result_df['list_r_square'].iloc[0],
)
train_loss = result_df['list_train_loss'].iloc[0]
logS_total = result_df['logS_total'].iloc[0]
pred_logS_total = result_df['pred_logS_total'].iloc[0]

plot_loss_curve(train_loss, save_prefix + "_loss.png")
plot_metric_curve(metric_values, save_prefix + "_metric.png", test_interval=configs.test_interval)
plot_result_distribution(logS_total, pred_logS_total, save_prefix + "_distribution.png")
