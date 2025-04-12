from PyQt5 import QtWidgets, QtCore, QtGui
import sys
import json
import os

from api.requests import login
from api.requests import register
from api.requests import get_user_profile

TOKEN_FILE = "client/token.json"

def save_token(token):
    os.makedirs(os.path.dirname(TOKEN_FILE), exist_ok=True)
    with open(TOKEN_FILE, "w") as token_file:
        json.dump({"token": token}, token_file)
    print("Токен сохранен: ", token)

def load_token():
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "r") as token_file:
            data = json.load(token_file)
            return data.get("token")
    else:
        save_token("")
        return None

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
        self.setWindowTitle('Анализ временных рядов')
        self.setGeometry(100, 100, 800, 600)

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
    window = LoginWindow()
    window.show()
    sys.exit(app.exec_())