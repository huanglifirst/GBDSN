#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Bidirectional, Input, Concatenate, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score

# 读取数据
df = pd.read_csv('93-Site_DKA-M4_A-Phase.csv')
df = df.iloc[608250:815000]
idx_df = df.set_index(['timestamp'])

# 数据清洗
df2 = idx_df[['Active_Power', 'Wind_Speed', 'Weather_Temperature_Celsius', 'Weather_Relative_Humidity', 'Global_Horizontal_Radiation',
              'Diffuse_Horizontal_Radiation', 'Weather_Daily_Rainfall', 'Radiation_Global_Tilted']]
df2_cleaned = df2.dropna().query('Active_Power >= 0')

# 归一化
max_target = df2_cleaned['Active_Power'].max()
min_target = df2_cleaned['Active_Power'].min()
x = df2_cleaned.values
x_sc = MinMaxScaler()
x = x_sc.fit_transform(x)

# 滑动窗口截取数据
def multivariate_data(dataset, target, start_index, end_index, history_size, target_size, step, single_step=False):
    data, labels = [], []
    start_index += history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])
        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])

    return np.array(data), np.array(labels)

train_ratio, val_ratio, test_ratio = 0.7, 0.2, 0.1
total_length = len(x)
train_end_index = int(total_length * train_ratio)
val_end_index = train_end_index + int(total_length * val_ratio)
past_history, future_target, STEP = 12, 3, 1  # 修改 future_target 为 3 以实现三步预测

x_train_multi, y_train_multi = multivariate_data(x, x[:, 0], 0, train_end_index, past_history, future_target, STEP, single_step=False)
x_val_multi, y_val_multi = multivariate_data(x, x[:, 0], train_end_index, val_end_index, past_history, future_target, STEP, single_step=False)
x_test_multi, y_test_multi = multivariate_data(x, x[:, 0], val_end_index, None, past_history, future_target, STEP, single_step=False)

# GCN-BiLSTM 模型定义
class NewGraphConvolution(tf.keras.layers.Layer):
    def __init__(self, output_dim, adj_matrix, dropout_rate=0.5, l2_reg=0, activation='relu', seed=1024, **kwargs):
        self.output_dim = output_dim
        self.adj_matrix = adj_matrix
        self.activation = tf.keras.activations.get(activation)
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.seed = seed
        super(NewGraphConvolution, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', shape=(input_shape[-1], self.output_dim),
                                      initializer='glorot_uniform', regularizer=tf.keras.regularizers.l2(self.l2_reg), trainable=True)
        self.bias = self.add_weight(name='bias', shape=(self.output_dim,), initializer='zeros', trainable=True)
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate, seed=self.seed)
        super(NewGraphConvolution, self).build(input_shape)

    def call(self, x):
        adj_matrix = tf.cast(self.adj_matrix, dtype=tf.float32)
        output = tf.matmul(adj_matrix, x)
        output = tf.matmul(output, self.kernel) + self.bias
        output = self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = super(NewGraphConvolution, self).get_config()
        config.update({
            'output_dim': self.output_dim,
            'adj_matrix': self.adj_matrix.tolist(),
            'activation': self.activation,
            'dropout_rate': self.dropout_rate,
            'l2_reg': self.l2_reg,
            'seed': self.seed
        })
        return config

    @classmethod
    def from_config(cls, config):
        config['adj_matrix'] = tf.constant(config['adj_matrix'])
        return cls(**config)

tf.keras.utils.get_custom_objects().update({'NewGraphConvolution': NewGraphConvolution})

def build_gcn_bilstm_model(adj_matrix, gcn_output_dim, lstm_units, input_shape, dropout_rate=0.5, l2_reg=1e-4):
    gcn_input = Input(shape=(input_shape[1], input_shape[2]), name='gcn_input')
    gcn_layer = NewGraphConvolution(output_dim=gcn_output_dim, adj_matrix=adj_matrix, dropout_rate=dropout_rate, l2_reg=l2_reg)(gcn_input)
    gcn_layer = Dense(units=32, activation='relu')(gcn_layer)
    gcn_output = tf.reduce_mean(gcn_layer, axis=1)

    lstm_input = Input(shape=(input_shape[1], input_shape[2]), name='lstm_input')
    bilstm_layer = Bidirectional(LSTM(units=lstm_units, return_sequences=False, dropout=dropout_rate, recurrent_dropout=dropout_rate))(lstm_input)

    combined = Concatenate()([gcn_output, bilstm_layer])
    dense_layer = Dense(units=64, activation='relu')(combined)
    dense_layer = Dropout(dropout_rate)(dense_layer)
    output = Dense(units=future_target, activation='linear')(dense_layer)  # 修改输出层以输出未来3个时间步的预测值

    model = Model(inputs=[gcn_input, lstm_input], outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# 计算相关性矩阵并生成邻接矩阵
corr_matrix = df2_cleaned.corr().round(3)
sorted_corr = np.sort(corr_matrix.values.ravel())[::-1]
threshold = sorted_corr[int(len(sorted_corr) * 0.4)]
adj_matrix = corr_matrix.where(corr_matrix >= threshold, 0)
adj_matrix.to_csv('adjacency_matrix3.csv')

# 读取邻接矩阵文件并删除第一列
adj = pd.read_csv('adjacency_matrix3.csv')
adj = adj.drop(columns=[adj.columns[0]])
adj.to_csv('adjacency_matrix3.csv', index=False)

# 构建模型并绘制模型结构
adj_matrix = adj.values
input_shape = (None, 8, 12)
gcn_output_dim = 16
lstm_units = 64
model = build_gcn_bilstm_model(adj_matrix, gcn_output_dim, lstm_units, input_shape)
model.summary()
plot_model(model, to_file='model_structure.png', show_shapes=True, show_layer_names=True)

# 模型训练
x_train_multi_transposed = np.transpose(x_train_multi, (0, 2, 1))
checkpoint_filepath = '3_step_multi_best_model.h5'
model_checkpoint_callback = ModelCheckpoint(filepath=checkpoint_filepath, save_best_only=True, monitor='val_loss', mode='min', verbose=1)

history = model.fit([x_train_multi_transposed, x_train_multi_transposed], y_train_multi, batch_size=32, epochs=100, validation_split=0.2, shuffle=True, callbacks=[model_checkpoint_callback])

plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

# 加载最佳模型并进行预测
best_model = tf.keras.models.load_model(checkpoint_filepath, custom_objects={'NewGraphConvolution': NewGraphConvolution})
x_test_multi_transposed = np.transpose(x_test_multi, (0, 2, 1))
y_pred = best_model.predict([x_test_multi_transposed, x_test_multi_transposed])

# 反归一化
y_pred_unnormalized = y_pred * (max_target - min_target) + min_target
y_test_multi_unnormalized = y_test_multi * (max_target - min_target) + min_target
y_pred_original = y_pred_unnormalized
y_test_original = y_test_multi_unnormalized

# 计算误差
mse = mean_squared_error(y_test_original, y_pred_original)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_original, y_pred_original)
mbe = np.mean(y_test_original - y_pred_original)
nrmse = rmse / (np.max(y_test_original) - np.min(y_test_original))
r2 = r2_score(y_test_original, y_pred_original)
evs = explained_variance_score(y_test_original, y_pred_original)

print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
print("Mean Bias Error(MBE):", mbe)
print("NRMSE:", nrmse)
print("Explained Variance Score (EVS):", evs)
print("R² Score:", r2)

# 绘制预测结果
plt.figure(figsize=(25, 10))
plt.plot(y_test_original.flatten(), label='Actual', color='blue', linewidth=0.5)
plt.plot(y_pred_original.flatten(), label='Predicted', color='red', linewidth=0.5)
plt.legend()
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Actual vs. Predicted Data')
plt.savefig('actual_vs_predicted_full.png')  # 保存完整预测结果图像
plt.show()

plt.figure(figsize=(25, 10))
plt.plot(y_test_original.flatten()[500:5000], label='Actual', color='blue', linewidth=0.5)
plt.plot(y_pred_original.flatten()[500:5000], label='Predicted', color='red', linewidth=0.5)
plt.legend()
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Actual vs. Predicted Data')
plt.savefig('actual_vs_predicted_zoomed.png')  # 保存局部预测结果图像
plt.show()


"""
Mean Squared Error (MSE): 0.0935135431663275
Root Mean Squared Error (RMSE): 0.3057998416715213
Mean Absolute Error (MAE): 0.1543773782588919
Mean Bias Error(MBE): 0.011242752618148544
NRMSE: 0.057711882375993825
Explained Variance Score (EVS): 0.9581515790284819
R² Score: 0.958085628910363
"""