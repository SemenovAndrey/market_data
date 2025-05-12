from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
from pyqtgraph import PlotWidget, DateAxisItem
import sys
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# from client.ui.main_window import MainWindow
# from client.ui.login_window import LoginWindow

# from tensorflow.python.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense, LSTM

# from keras.src.models import Sequential
# from keras.src.layers import LSTM, Dense

# from keras.src.models import Sequential
# from keras.src.layers import LSTM, Dense
# from keras.src.models.sequential import Sequential
# from keras.src.layers.rnn.lstm import LSTM
# from keras.src.layers.core.dense import Dense
# from keras.src.layers.rnn import rnn

from utils.token_functions import load_token, save_token, TOKEN_FILE

from api.requests import login, register, get_asset_history, get_user_profile
from api.requests import get_forecasting_methods, get_indicator_names


if __name__ == '__main__':
    from client.ui.main_window import MainWindow

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
