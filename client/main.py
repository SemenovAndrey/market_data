from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
from pyqtgraph import PlotWidget, DateAxisItem
import sys
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

from utils.token_functions import load_token, save_token, TOKEN_FILE
# from utils.graph_functions import draw_graph

from api.requests import login, register, get_asset_history, get_user_profile

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
        self.close()
        self.register_window = RegisterWindow()
        self.register_window.show()

    def open_main_window(self):
        self.close()
        self.main_window = MainWindow()
        self.main_window.show()

class RegisterWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.main_window = None
        self.login_window = None

        self.setWindowTitle("Регистрация")
        self.setGeometry(100, 100, 600, 400)

        self.title_label = QtWidgets.QLabel("Регистрация")
        self.title_label.setAlignment(QtCore.Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 24px; font-weight: bold; padding: 10px;")

        self.username_label = QtWidgets.QLabel("Логин")
        self.username_label.setStyleSheet("font-size: 16px; margin-bottom: 5px;")
        self.username_input = QtWidgets.QLineEdit()
        self.username_input.setPlaceholderText("Введите логин")
        self.username_input.setFixedHeight(40)
        self.username_input.setStyleSheet("font-size: 16px; padding: 5px; border-radius: 5px;")

        self.email_label = QtWidgets.QLabel("Почта")
        self.email_label.setStyleSheet("font-size: 16px; margin-bottom: 5px;")
        self.email_input = QtWidgets.QLineEdit()
        self.email_input.setPlaceholderText("Введите почту")
        self.email_input.setFixedHeight(40)
        self.email_input.setStyleSheet("font-size: 16px; padding: 5px; border-radius: 5px;")

        self.password_label = QtWidgets.QLabel("Пароль")
        self.password_label.setStyleSheet("font-size: 16px; margin-bottom: 5px;")
        self.password_input = QtWidgets.QLineEdit()
        self.password_input.setEchoMode(QtWidgets.QLineEdit.Password)
        self.password_input.setPlaceholderText("Введите пароль")
        self.password_input.setFixedHeight(40)
        self.password_input.setStyleSheet("font-size: 16px; padding: 5px; border-radius: 5px;")

        self.login_button = QtWidgets.QPushButton("Зарегистрироваться")
        self.login_button.setFixedHeight(45)
        self.login_button.setStyleSheet("""
                    font-size: 18px;
                    background-color: #4CAF50;
                    color: white;
                    border-radius: 5px;
                    padding: 5px;
                """)
        self.login_button.clicked.connect(self.register)

        self.register_button = QtWidgets.QPushButton("Войти")
        self.register_button.setFixedHeight(45)
        self.register_button.setStyleSheet("""
                   font-size: 18px;
                   background-color: #2196F3;
                   color: white;
                   border-radius: 5px;
                   padding: 5px;
               """)
        self.register_button.clicked.connect(self.open_login_window)

        layout_username = QtWidgets.QVBoxLayout()
        layout_username.addWidget(self.username_label)
        layout_username.addWidget(self.username_input)

        layout_email = QtWidgets.QVBoxLayout()
        layout_email.addWidget(self.email_label)
        layout_email.addWidget(self.email_input)

        layout_password = QtWidgets.QVBoxLayout()
        layout_password.addWidget(self.password_label)
        layout_password.addWidget(self.password_input)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.title_label)
        layout.addLayout(layout_username)
        layout.addLayout(layout_email)
        layout.addLayout(layout_password)
        layout.addWidget(self.login_button)
        layout.addWidget(self.register_button)

        self.setLayout(layout)

    def register(self):
        username = self.username_input.text()
        email = self.email_input.text()
        password = self.password_input.text()

        token = register(username, email, password)
        if token:
            save_token(token)
            QtWidgets.QMessageBox.information(self, "Успешно", "Регистрация выполнена")
            self.open_main_window()
        else:
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Некорректный ввод данных")

    def open_login_window(self):
        self.close()
        self.login_window = LoginWindow()
        self.login_window.show()

    def open_main_window(self):
        self.close()
        self.main_window = MainWindow()
        self.main_window.show()

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.last_graph_data = None

        self.setWindowTitle('Анализ графиков')
        self.setGeometry(100, 100, 1200, 700)

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout()
        central_widget.setLayout(main_layout)

        header_layout = QtWidgets.QHBoxLayout()

        self.title_label = QtWidgets.QLabel("Анализ графиков")
        self.title_label.setAlignment(QtCore.Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 24px; font-weight: bold;")

        header_spacer = QtWidgets.QWidget()
        header_spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)

        self.profile_button = QtWidgets.QPushButton("Профиль")
        self.logout_button = QtWidgets.QPushButton("Выход")

        header_layout.addWidget(self.title_label)
        header_layout.addWidget(header_spacer)
        header_layout.addWidget(self.profile_button)
        header_layout.addWidget(self.logout_button)

        search_layout = QtWidgets.QHBoxLayout()
        self.search_input = QtWidgets.QLineEdit()
        self.search_input.setPlaceholderText("Введите тикер актива (например: AAPL, BTC-USD)")
        self.search_button = QtWidgets.QPushButton("Показать график")
        self.search_button.clicked.connect(self.draw_graph)

        search_layout.addWidget(self.search_input)
        search_layout.addWidget(self.search_button)

        content_layout = QtWidgets.QHBoxLayout()

        axis = DateAxisItem(orientation='bottom')
        self.graph_area = pg.PlotWidget(axisItems={'bottom': axis})
        self.graph_area.setBackground("w")
        self.graph_area.setTitle("График актива", color="k", size="16pt")
        self.graph_area.setLabel("left", "Цена", color="k", size="14pt")
        self.graph_area.setLabel("bottom", "Время", color="k", size="14pt")
        self.graph_area.showGrid(x=True, y=True)
        self.graph_area.setMinimumSize(1000, 500)
        self.graph_area.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        right_panel = QtWidgets.QVBoxLayout()

        self.indicator_label = QtWidgets.QLabel("Выберите индикатор:")
        self.indicator_combo = QtWidgets.QComboBox()
        self.indicator_combo.addItems(["SMA", "EMA", "RSI", "MACD"])
        self.indicator_button = QtWidgets.QPushButton("Вывести индикатор")
        self.indicator_clear_button = QtWidgets.QPushButton("Очистить индикатор")

        self.model_label = QtWidgets.QLabel("Выберите модель прогноза:")
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.addItems(["SMA-прогноз", "ARIMA", "Linear Regression", "LSTM (позже)"])
        self.model_button = QtWidgets.QPushButton("Отобразить прогноз")
        self.model_button.clicked.connect(self.draw_arima_forecast)
        self.model_clear_button = QtWidgets.QPushButton("Очистить прогноз")

        right_panel.addWidget(self.indicator_label)
        right_panel.addWidget(self.indicator_combo)
        right_panel.addWidget(self.indicator_button)
        right_panel.addWidget(self.indicator_clear_button)
        right_panel.addSpacing(30)
        right_panel.addWidget(self.model_label)
        right_panel.addWidget(self.model_combo)
        right_panel.addWidget(self.model_button)
        right_panel.addWidget(self.model_clear_button)
        right_panel.addStretch()

        content_layout.addWidget(self.graph_area, 3)
        content_layout.addLayout(right_panel, 1)

        main_layout.addLayout(header_layout)
        main_layout.addSpacing(10)
        main_layout.addLayout(search_layout)
        main_layout.addSpacing(20)
        main_layout.addLayout(content_layout)

    def draw_graph(self):
        symbol = self.search_input.text().strip()
        if not symbol:
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Введите тикер актива")
            return

        data = get_asset_history(symbol)
        if not data:
            QtWidgets.QMessageBox.warning(self, "Ошибка", f"Нет или недостаточно данных для {symbol}")
            return

        x = [datetime.fromisoformat(point["timestamp"]).timestamp() for point in data]
        y = [point["close"] for point in data]

        self.last_graph_data = {"x": x, "y": y}

        self.graph_area.clear()
        self.graph_area.plot(
            x,
            y,
            pen=pg.mkPen(color="b", width=2)
        )

        self.graph_area.setTitle(f"График {symbol.upper()}", color="k", size="16pt")

    def predict_arima(self, y_values, last_x):
        if len(y_values) < 30:
            QtWidgets.QMessageBox.warning(self, "Недостаточно данных", "Для прогноза необходимо минимум 30 точек.")
            return [], []

        try:
            model = ARIMA(y_values, order=(2, 1, 2))
            model_fit = model.fit()

            forecast = model_fit.forecast(steps=5)
            last_date = last_x[-1]

            # Генерация новых дат
            interval_sec = last_x[-1] - last_x[-2]  # разница между точками
            x_pred = [last_date + (i + 1) * interval_sec for i in range(5)]

            return x_pred, forecast.tolist()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка ARIMA", str(e))
            return [], []

    def draw_arima_forecast(self):
        if not hasattr(self, "last_graph_data"):
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Сначала отобразите график.")
            return

        x = self.last_graph_data["x"]
        y = self.last_graph_data["y"]

        x_pred, y_pred = self.predict_arima(y, x)
        if x_pred and y_pred:
            self.graph_area.plot(
                x_pred,
                y_pred,
                pen=pg.mkPen(color="g", style=QtCore.Qt.DashLine, width=2),
                name="Прогноз"
            )

class ProfileWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Профиль пользователя")
        self.setGeometry(100, 100, 300, 150)

        self.profile_label = QtWidgets.QLabel("Загрузка профиля...")
        self.refresh_button = QtWidgets.QPushButton("Обновить")
        self.refresh_button.clicked.connect(self.load_profile)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.profile_label)
        layout.addWidget(self.refresh_button)
        self.setLayout(layout)

        self.load_profile()

    def load_profile(self):
        profile = get_user_profile()
        if profile:
            self.profile_label.setText(profile.get("message", "Профиль не найден"))
        else:
            self.profile_label.setText("Ошибка загрузки профиля")

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
