# from PyQt5 import QtWidgets, QtCore, QtGui
# import pyqtgraph as pg
# from pyqtgraph import PlotWidget, DateAxisItem
# import sys
# from datetime import datetime
# from statsmodels.tsa.arima.model import ARIMA
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# from sklearn.preprocessing import MinMaxScaler
#
# # from client.ui.profile_window import ProfileWindow
# # from client.ui.login_window import LoginWindow
#
# # from tensorflow.python.keras.models import Sequential
# # from tensorflow.python.keras.layers import Dense, LSTM
#
# # from keras.src.models import Sequential
# # from keras.src.layers import LSTM, Dense
#
# # from keras.src.models import Sequential
# # from keras.src.layers import LSTM, Dense
# # from keras.src.models.sequential import Sequential
# # from keras.src.layers.rnn.lstm import LSTM
# # from keras.src.layers.core.dense import Dense
# # from keras.src.layers.rnn import rnn
#
# from client.utils.token_functions import load_token, save_token, TOKEN_FILE
#
# from client.api.requests import get_asset_history
# from client.api.requests import get_forecasting_methods, get_indicator_names, get_indicator_data
# from client.charts.indicator import calculate_sma, calculate_ema, calculate_rsi, calculate_macd
#
#
# def predict_arima(self, y_values, last_x):
#     if len(y_values) < 30:
#         QtWidgets.QMessageBox.warning(self, "Недостаточно данных", "Для прогноза необходимо минимум 30 точек.")
#         return [], []
#
#     try:
#         model = ARIMA(y_values, order=(2, 1, 2))
#         # model = ARIMA(y_values, order=(3, 1, 0))
#         # model = ARIMA(y_values, order=(1, 1, 1))
#         model_fit = model.fit()
#
#         forecast = model_fit.forecast(steps=5)
#         last_date = last_x[-1]
#
#         interval_sec = last_x[-1] - last_x[-2]
#         x_pred = [last_date + (i + 1) * interval_sec for i in range(5)]
#
#         return x_pred, forecast.tolist()
#     except Exception as e:
#         QtWidgets.QMessageBox.critical(self, "Ошибка ARIMA", str(e))
#         return [], []
#
# # ------------------------------------------------------------------------------------------------------------------
# # LSTM TENSORFLOW
# # ------------------------------------------------------------------------------------------------------------------
# # def predict_lstm(self, y_values, x_values):
# #     if len(y_values) < 60:
# #         QtWidgets.QMessageBox.warning(self, "Недостаточно данных", "Для LSTM необходимо минимум 60 точек.")
# #         return [], []
# #
# #     scaler = MinMaxScaler()
# #     y_scaled = scaler.fit_transform(np.array(y_values).reshape(-1, 1))
# #
# #     x_train = []
# #     y_train = []
# #     for i in range(60, len(y_scaled)):
# #         x_train.append(y_scaled[i - 60:i, 0])
# #         y_train.append(y_scaled[i, 0])
# #
# #     x_train = np.array(x_train)
# #     y_train = np.array(y_train)
# #     x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
# #
# #     model = Sequential()
# #     model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
# #     model.add(LSTM(units=50))
# #     model.add(Dense(1))
# #     model.compile(optimizer='adam', loss='mean_squared_error')
# #     model.fit(x_train, y_train, epochs=5, batch_size=16, verbose=0)
# #
# #     input_seq = y_scaled[-60:].reshape(1, 60, 1)
# #     y_pred = []
# #     for _ in range(5):
# #         next_val = model.predict(input_seq, verbose=0)[0][0]
# #         y_pred.append(next_val)
# #         input_seq = np.append(input_seq[:, 1:, :], [[[next_val]]], axis=1)
# #
# #     y_pred_rescaled = scaler.inverse_transform(np.array(y_pred).reshape(-1, 1)).flatten()
# #
# #     interval = x_values[-1] - x_values[-2]
# #     x_pred = [x_values[-1] + (i + 1) * interval for i in range(5)]
# # ------------------------------------------------------------------------------------------------------------------
#
# # ------------------------------------------------------------------------------------------------------------------
# # LSTM TORCH v1.1
# # ------------------------------------------------------------------------------------------------------------------
# # def predict_lstm(self, y_values, x_values):
# #     if len(y_values) < 60:
# #         QtWidgets.QMessageBox.warning(self, "Недостаточно данных", "Для LSTM необходимо минимум 60 точек.")
# #         return [], []
# #
# #     class LSTMModel(nn.Module):
# #         def __init__(self, input_size=1, hidden_size=50, num_layers=2):
# #             super(LSTMModel, self).__init__()
# #             self.hidden_size = hidden_size
# #             self.num_layers = num_layers
# #             self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
# #             self.fc = nn.Linear(hidden_size, 1)
# #
# #         def forward(self, x, hidden):
# #             out, hidden = self.lstm(x, hidden)
# #             out = self.fc(out[:, -1, :])
# #             return out, hidden
# #
# #         def init_hidden(self, batch_size):
# #             # инициализируем hidden и cell состояния
# #             return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
# #                     torch.zeros(self.num_layers, batch_size, self.hidden_size))
# #
# #     # Нормализация
# #     scaler = MinMaxScaler()
# #     y_scaled = scaler.fit_transform(np.array(y_values).reshape(-1, 1))
# #
# #     # Формируем обучающие примеры
# #     x_train = np.array([y_scaled[i - 60:i, 0] for i in range(60, len(y_scaled))])
# #     y_train = np.array([y_scaled[i, 0] for i in range(60, len(y_scaled))])
# #
# #     x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(2)  # (samples, 60, 1)
# #     y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # (samples, 1)
# #
# #     # Модель
# #     model = LSTMModel()
# #     criterion = nn.MSELoss()
# #     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# #
# #     # Обучение
# #     model.train()
# #     for epoch in range(90):
# #         hidden = model.init_hidden(x_train.size(0))  # <-- Сюда size батча
# #         output, hidden = model(x_train, hidden)  # <-- Передаём hidden
# #         loss = criterion(output, y_train)
# #         optimizer.zero_grad()
# #         loss.backward()
# #         optimizer.step()
# #
# #     # Прогноз
# #     model.eval()
# #     input_seq = torch.tensor(y_scaled[-60:], dtype=torch.float32).unsqueeze(0)#.unsqueeze(2)  # (1, 60, 1)
# #     hidden = model.init_hidden(1)
# #
# #     y_pred = []
# #     for _ in range(5):
# #         with torch.no_grad():
# #             print(f"input_seq.shape: {input_seq.shape}")
# #
# #             next_val, hidden = model(input_seq, hidden)
# #             next_val_scalar = next_val.item()
# #             y_pred.append(next_val_scalar)
# #
# #             # next_tensor = torch.tensor([[[next_val_scalar]]], dtype=torch.float32)
# #             # input_seq = torch.cat((input_seq[:, 1:, :], next_tensor.unsqueeze(0)), dim=1)
# #             next_tensor = torch.tensor([[[next_val_scalar]]], dtype=torch.float32)  # (1,1,1) — без .unsqueeze(0) дополнительно!
# #             input_seq = torch.cat((input_seq[:, 1:, :], next_tensor), dim=1)  # теперь размерность сохранится (1,60,1)
# #
# #     # Обратное масштабирование
# #     y_pred_rescaled = scaler.inverse_transform(np.array(y_pred).reshape(-1, 1)).flatten()
# #
# #     interval = x_values[-1] - x_values[-2]
# #     x_pred = [x_values[-1] + (i + 1) * interval for i in range(5)]
# #
# #     return x_pred, y_pred_rescaled.tolist()
# # ------------------------------------------------------------------------------------------------------------------
#
# # ------------------------------------------------------------------------------------------------------------------
# # LSTM TORCH v1.2
# # ------------------------------------------------------------------------------------------------------------------
# def predict_lstm(self, y_values, x_values):
#     if len(y_values) < 60:
#         QtWidgets.QMessageBox.warning(self, "Недостаточно данных", "Для LSTM необходимо минимум 60 точек.")
#         return [], []
#
#     class LSTMModel(nn.Module):
#         def __init__(self, input_size=1, hidden_size=64, num_layers=2):
#             super(LSTMModel, self).__init__()
#             self.hidden_size = hidden_size
#             self.num_layers = num_layers
#             self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
#             self.fc = nn.Linear(hidden_size, 1)
#
#         def forward(self, x, hidden):
#             out, hidden = self.lstm(x, hidden)
#             out = self.fc(out[:, -1, :])
#             return out, hidden
#
#         def init_hidden(self, batch_size):
#             return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
#                     torch.zeros(self.num_layers, batch_size, self.hidden_size))
#
#     # Нормализация данных
#     scaler = MinMaxScaler()
#     y_scaled = scaler.fit_transform(np.array(y_values).reshape(-1, 1))
#
#     # Формирование обучающей выборки
#     x_train = np.array([y_scaled[i - 60:i, 0] for i in range(60, len(y_scaled))])
#     y_train = np.array([y_scaled[i, 0] for i in range(60, len(y_scaled))])
#
#     x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(-1)  # (samples, 60, 1)
#     y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # (samples, 1)
#
#     model = LSTMModel()
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#
#     # Обучение модели
#     model.train()
#     for epoch in range(200):
#         hidden = model.init_hidden(x_train.size(0))
#         output, _ = model(x_train, hidden)
#         loss = criterion(output, y_train)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#     # Прогнозирование
#     model.eval()
#     input_seq = torch.tensor(y_scaled[-60:].reshape(1, 60, 1), dtype=torch.float32)#.unsqueeze(0)#.unsqueeze(-1)  # (1, 60, 1)
#     hidden = model.init_hidden(1)
#
#     y_pred = []
#     for _ in range(5):
#         with torch.no_grad():
#             print(f"input_seq.shape: {input_seq.shape}")
#
#             next_val, hidden = model(input_seq, hidden)
#             next_val_scalar = next_val.item()
#             y_pred.append(next_val_scalar)
#
#             next_tensor = torch.tensor([[[next_val_scalar]]], dtype=torch.float32)
#             input_seq = torch.cat((input_seq[:, 1:, :], next_tensor), dim=1)
#
#     # Обратное масштабирование
#     y_pred_rescaled = scaler.inverse_transform(np.array(y_pred).reshape(-1, 1)).flatten()
#
#     # Формирование временных меток для прогноза
#     interval = x_values[-1] - x_values[-2]
#     x_pred = [x_values[-1] + (i + 1) * interval for i in range(5)]
#
#     return x_pred, y_pred_rescaled.tolist()
# # ------------------------------------------------------------------------------------------------------------------