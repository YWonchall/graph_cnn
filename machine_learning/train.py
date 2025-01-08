import pandas as pd
import machine_learning.config as configs
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from machine_learning.utils import extract_feature
import xgboost as xgb
from sklearn.svm import SVR

# 读取训练数据和测试数据
df_train = pd.read_csv(configs.train_path)
df_train = pd.concat([df_train[configs.input_name], df_train[configs.label_name]], axis=1)
df_train.columns = [configs.input_name,  configs.label_name]
df_train = df_train.dropna(axis=0).reset_index(drop=True)

df_val = pd.read_csv(configs.val_path)
df_val = pd.concat([df_val[configs.input_name], df_val[configs.label_name]], axis=1)
df_val.columns = [configs.input_name,  configs.label_name]
df_val = df_val.dropna(axis=0).reset_index(drop=True)

# 假设CSV中有特征列和目标列，特征列为'feature1', 'feature2', ...，目标列为'target'
# 分离特征和目标变量
X_train = extract_feature(df_train[configs.input_name])
y_train = df_train[configs.label_name]

X_val = extract_feature(df_val[configs.input_name])
y_val = df_val[configs.label_name]

print('- Train Data :', len(X_train))
print('- Test Data :', len(X_val))

model = xgb.XGBRegressor()
# model = RandomForestRegressor()
# model = SVR()

# 训练模型
model.fit(X_train, y_train)

# 在测试集上预测
y_pred = model.predict(X_val)

# 评估模型性能
mae = mean_absolute_error(y_val, y_pred)
mse = mean_squared_error(y_val, y_pred)
r_square = r2_score(y_val, y_pred)

print('[Test]')
print('- MAE : %.4f' % mae)
print('- MSE : %.4f' % mse)
print('- R2 : %.4f' % r_square)

