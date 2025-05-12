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

from client.utils.token_functions import load_token, save_token, TOKEN_FILE

from client.api.requests import login, register, get_asset_history, get_user_profile
from client.api.requests import get_forecasting_methods, get_indicator_names

class LoginWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.main_window = None
        self.register_window = None

        self.setWindowTitle("Аутентификация")
        self.setGeometry(100, 100, 600, 400)

        self.title_label = QtWidgets.QLabel("Аутентификация")
        self.title_label.setAlignment(QtCore.Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 24px; font-weight: bold; padding: 10px;")

        self.username_label = QtWidgets.QLabel("Логин")
        self.username_label.setStyleSheet("font-size: 16px; margin-bottom: 5px;")
        self.username_input = QtWidgets.QLineEdit()
        self.username_input.setPlaceholderText("Введите логин")
        self.username_input.setFixedHeight(40)
        self.username_input.setStyleSheet("font-size: 16px; padding: 5px; border-radius: 5px;")

        self.password_label = QtWidgets.QLabel("Пароль")
        self.password_label.setStyleSheet("font-size: 16px; margin-bottom: 5px;")
        self.password_input = QtWidgets.QLineEdit()
        self.password_input.setEchoMode(QtWidgets.QLineEdit.Password)
        self.password_input.setPlaceholderText("Введите пароль")
        self.password_input.setFixedHeight(40)
        self.password_input.setStyleSheet("font-size: 16px; padding: 5px; border-radius: 5px;")

        self.login_button = QtWidgets.QPushButton("Войти")
        self.login_button.setFixedHeight(45)
        self.login_button.setStyleSheet("""
                    font-size: 18px;
                    background-color: #4CAF50;
                    color: white;
                    border-radius: 5px;
                    padding: 5px;
                """)
        self.login_button.clicked.connect(self.login)

        self.register_button = QtWidgets.QPushButton("Зарегистрироваться")
        self.register_button.setFixedHeight(45)
        self.register_button.setStyleSheet("""
                   font-size: 18px;
                   background-color: #2196F3;
                   color: white;
                   border-radius: 5px;
                   padding: 5px;
               """)
        self.register_button.clicked.connect(self.open_register_window)

        layout_username = QtWidgets.QVBoxLayout()
        layout_username.addWidget(self.username_label)
        layout_username.addWidget(self.username_input)

        layout_password = QtWidgets.QVBoxLayout()
        layout_password.addWidget(self.password_label)
        layout_password.addWidget(self.password_input)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.title_label)
        layout.addLayout(layout_username)
        layout.addLayout(layout_password)
        layout.addWidget(self.login_button)
        layout.addWidget(self.register_button)

        self.setLayout(layout)

    def login(self):
        username = self.username_input.text()
        password = self.password_input.text()

        token = login(username, password)
        if token:
            save_token(token)
            QtWidgets.QMessageBox.information(self, "Успешно", "Вход выполнен")
            self.open_main_window()
        else:
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Неверный логин или пароль")

    def open_register_window(self):
        from client.ui.register_window import RegisterWindow

        self.close()
        self.register_window = RegisterWindow()
        self.register_window.show()

    def open_main_window(self):
        from client.ui.main_window import MainWindow

        self.close()
        self.main_window = MainWindow()
        self.main_window.show()