import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import torch
from torch_geometric.data import DataLoader
import copy
import warnings
import os


from smiles_to_graph import make_regre_mol
from smiles_to_graph import make_regre_vec
from runner import experiment
import model

warnings.filterwarnings(action='ignore')
parser = argparse.ArgumentParser(
    description="Graph Convolutional Network for logS Regression", 
    epilog="python regression.py -D './myDataset.xlsx' -X1 'Solute SMILES'  -Y 'LogS' -O './results/myResult.json' -M './results/myModel.pt'")
parser.add_argument('--seed', '-s', type=int, default=712, help='seed')
parser.add_argument('--train_path', '-T', type=str, required=True, help="dataset path and name ('./dataset.xlsx')")
parser.add_argument('--val_path', '-V', type=str, required=True, help="dataset path and name ('./dataset.xlsx')")
parser.add_argument('--solute_smiles', '-X1', type=str, required=True, help="column name of solute smiles ('Solute SMILES')")
#parser.add_argument('--solvent_smiles', '-X2', type=str, required=True, help="column name of solvent smiles ('Solvent SMILES')")
parser.add_argument('--logS', '-Y', type=str, required=True, help="column name of logS ('LogS')")
parser.add_argument('--output_path', '-O', type=str, default='./results/', 
                    help="output path")
parser.add_argument('--conv', '-c', type=str, default='GCNConv', choices=['GCNConv', 'ARMAConv', 'SAGEConv'], 
                    help='GCNConv/ARMAConv/SAGEConv (defualt=GCNConv)')
parser.add_argument('--test_size', '-z', type=float, default=0.2, help='test size (defualt=0.2)')
parser.add_argument('--random_state', '-r', type=int, default=123, help='random state')
parser.add_argument('--batch_size', '-b', type=int, default=32, help='batch size (defualt=256)')
parser.add_argument('--epoch', '-e', type=int, default=30, help='epoch (defualt=200)')
parser.add_argument('--lr', '-l', type=float, default=0.005, help='learning rate (defualt=0.005)')
parser.add_argument('--step_size', '-t', type=int, default=5, help='step_size of lr_scheduler (defualt=5)')
parser.add_argument('--gamma', '-g', type=float, default=0.9, help='gamma of lr_scheduler (defualt=0.9)')
parser.add_argument('--dropout', '-d', type=float, default=0.1, help='dropout (defualt=0.1)')
parser.add_argument('--exp_name', '-n', type=str, default='default_exp', help='experiment name')
args = parser.parse_args()

os.makedirs(args.output_path, exist_ok=True)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


print()
print('Graph Convolutional Network for logS Regression')
print('Soongsil University, Seoul, South Korea')
print('Computational Science and Artificial Intelligence Lab')
print()
print('[Preparing Data]')
print('- Device :', device)
print()


df_train = pd.read_csv(args.train_path)
df_train = pd.concat([df_train[args.solute_smiles], df_train[args.logS]], axis=1)
df_train.columns = ['Solute SMILES',  'logS']
df_train = df_train.dropna(axis=0).reset_index(drop=True)

df_val = pd.read_csv(args.val_path)
df_val = pd.concat([df_val[args.solute_smiles], df_val[args.logS]], axis=1)
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
train_loader = DataLoader(train_X, batch_size=args.batch_size, shuffle=True, drop_last=False)
test_loader = DataLoader(test_X, batch_size=len(test_X), shuffle=True, drop_last=False)

model = model.Net(args)
model = model.to(device)

print()
dict_result = dict()
result = vars(experiment(model, train_loader, test_loader, device, args))
dict_result[args.exp_name] = copy.deepcopy(result)
result_df = pd.DataFrame(dict_result).transpose()
result_df.to_json(os.path.join(args.output_path, args.exp_name + ".json"), orient='table')


train_loss = result_df['list_train_loss'].iloc[0]
logS_total = result_df['logS_total'].iloc[0]
pred_logS_total = result_df['pred_logS_total'].iloc[0]

plt.rcParams["figure.figsize"] = (10, 6)
plt.suptitle(args.exp_name, fontsize=16)

plt.subplot(1, 2, 1)
plt.ylim([0, 10])
plt.plot([e for e in range(len(train_loss))], [float(t) for t in train_loss], label="train_loss", c='blue')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
mae_test = 'MAE : ' + str(round(float(result_df['mae'].iloc[0]), 2))
mse_test = 'MSE : ' + str(round(float(result_df['mse'].iloc[0]), 2))
r_test = 'R2 : ' + str(round(float(result_df['r_square'].iloc[0]), 2))
plt.text(0, -1.5, mae_test, fontsize=12)
plt.text(0, -2, mse_test, fontsize=12)
plt.text(0, -2.5, r_test, fontsize=12)

plt.subplot(1, 2, 2)
plt.scatter(logS_total, pred_logS_total, alpha=0.4)
plt.plot(logS_total, logS_total, alpha=0.4, color='black')
plt.xlabel("logS_total")
plt.ylabel("pred_logS_total")

plt.tight_layout()
plt.savefig(os.path.join(args.output_path, args.exp_name + ".png"))
print()
