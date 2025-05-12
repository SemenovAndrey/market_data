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

from client.utils.token_functions import load_token, save_token, TOKEN_FILE

from client.api.requests import login, register, get_asset_history, get_user_profile
from client.api.requests import get_forecasting_methods, get_indicator_names


class ProfileWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.main_window = None
        self.login_window = None

        self.setWindowTitle("Профиль пользователя")
        self.setGeometry(100, 100, 300, 150)

        self.profile_label = QtWidgets.QLabel("Загрузка профиля...")
        self.refresh_button = QtWidgets.QPushButton("Обновить")
        self.refresh_button.clicked.connect(self.load_profile)
        self.bact_to_main_window_button = QtWidgets.QPushButton("Вернуться на главную")
        self.bact_to_main_window_button.clicked.connect(self.back_to_main_window)
        self.logout_button = QtWidgets.QPushButton("Выйти")
        self.logout_button.clicked.connect(self.logout)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.profile_label)
        layout.addWidget(self.refresh_button)
        layout.addWidget(self.bact_to_main_window_button)
        layout.addWidget(self.logout_button)
        self.setLayout(layout)

        self.load_profile()

    def load_profile(self):
        profile = get_user_profile()
        if profile:
            self.profile_label.setText(profile.get("message", "Профиль не найден"))
        else:
            self.profile_label.setText("Ошибка загрузки профиля")

    def back_to_main_window(self):
        from client.ui.main_window import MainWindow

        self.close()
        self.main_window = MainWindow()
        self.main_window.show()

    def logout(self):
        from client.ui.login_window import LoginWindow

        self.close()
        self.login_window = LoginWindow()
        self.login_window.show()