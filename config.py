# data
train_path = '/home/ywc/projects/graph_cnn/data/BIT/train.csv'
val_path = '/home/ywc/projects/graph_cnn/data/BIT/val.csv'
input_name = 'Solute SMILES'
label_name = 'LogS'
output_folder = './results/BIT'

# train val
pretrain_path = "/home/ywc/projects/graph_cnn/results/kinase_final_data/gnn_defult.pth"
batch_size = 64
epoch = 128
test_interval = 4
lr = 0.05
exp_name = "gnn_finetune_freeze"
seed = 712
test_size=0.2
random_state=123
step_size=5
gamma=0.9

# model
model_name="GNN"
conv='GCNConv'
dropout=0.1
