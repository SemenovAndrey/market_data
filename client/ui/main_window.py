from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
from PyQt5.QtCore import QDate
from PyQt5.QtWidgets import QPushButton
from pyqtgraph import PlotWidget, DateAxisItem
import sys
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# from client.ui.profile_window import ProfileWindow
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

# from client.utils.token_functions import load_token, save_token, TOKEN_FILE

from client.api.requests import get_asset_history
from client.api.requests import get_forecasting_methods, get_indicator_names, get_indicator_data
from client.charts.indicator import calculate_sma, calculate_ema, calculate_rsi, calculate_macd


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.login_window = None
        self.profile_window = None

        self.last_graph_data = None
        self.last_graph_data_xy = None
        self.current_symbol = None

        self.chart_mode = "line"

        self.indicator_panels = {}
        self.active_indicators = {}
        self.indicators_buttons = {}
        self.active_forecasts = {}
        self.forecasts_buttons = {}

        self.clear_button = None

        self.setWindowTitle('Анализ графиков')
        self.resize(1200, 700)

        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout()
        central_widget.setLayout(main_layout)

        # верхнее меню
        header_layout = QtWidgets.QHBoxLayout()

        self.title_label = QtWidgets.QLabel("Анализ графиков")
        self.title_label.setAlignment(QtCore.Qt.AlignCenter)
        self.title_label.setStyleSheet("font-size: 24px; font-weight: bold;")

        header_spacer = QtWidgets.QWidget()
        header_spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)

        self.profile_button = QtWidgets.QPushButton("Профиль")
        self.profile_button.clicked.connect(self.open_profile_window)
        self.logout_button = QtWidgets.QPushButton("Выход")
        self.logout_button.clicked.connect(self.open_login_window)

        header_layout.addWidget(self.title_label)
        header_layout.addWidget(header_spacer)
        header_layout.addWidget(self.profile_button)
        header_layout.addWidget(self.logout_button)

        search_layout = QtWidgets.QHBoxLayout()
        self.search_input = QtWidgets.QLineEdit()
        self.search_input.setPlaceholderText("Введите тикер актива (например: AAPL, BTC-USD, LKOH)")
        self.search_button = QtWidgets.QPushButton("Показать график")
        self.search_button.clicked.connect(self.get_data)

        self.chart_toggle_button = QtWidgets.QPushButton("Переключить вид графика")
        self.chart_toggle_button.clicked.connect(self.toggle_chart_mode)

        search_layout.addWidget(self.search_input)
        search_layout.addWidget(self.search_button)
        search_layout.addWidget(self.chart_toggle_button)

        self.period_checkbox = QtWidgets.QCheckBox("Выбрать период")
        self.start_date_edit = QtWidgets.QDateEdit()
        self.start_date_edit.setCalendarPopup(True)
        self.start_date_edit.setDisplayFormat("yyyy-MM-dd")
        self.start_date_edit.setEnabled(False)

        self.end_date_edit = QtWidgets.QDateEdit()
        self.end_date_edit.setCalendarPopup(True)
        self.end_date_edit.setDisplayFormat("yyyy-MM-dd")
        self.end_date_edit.setEnabled(False)

        self.period_checkbox.stateChanged.connect(self.toggle_date_inputs)

        search_layout.addWidget(self.period_checkbox)
        search_layout.addWidget(self.start_date_edit)
        search_layout.addWidget(self.end_date_edit)

        content_layout = QtWidgets.QHBoxLayout()

        # графики
        axis = DateAxisItem(orientation='bottom')

        self.graph_area = pg.PlotWidget(axisItems={'bottom': axis})
        self.graph_area.setBackground("w")
        self.graph_area.setTitle("График актива", color="k", size="16pt")
        self.graph_area.setLabel("left", "Цена", color="k", size="14pt")
        self.graph_area.setLabel("bottom", "Время", color="k", size="14pt")
        self.graph_area.showGrid(x=True, y=True)
        self.graph_area.setMinimumHeight(300)

        self.volume_area = pg.PlotWidget()
        self.volume_area.setMaximumHeight(100)
        self.volume_area.setBackground("w")
        self.volume_area.setLabel("left", "Объем", color="k", size="10pt")
        self.volume_area.showGrid(x=True, y=True)
        self.volume_area.setXLink(self.graph_area)

        self.rsi_area = pg.PlotWidget()
        self.rsi_area.setMaximumHeight(100)
        self.rsi_area.setBackground("w")
        self.rsi_area.setLabel("left", "RSI", color="k", size="10pt")
        self.rsi_area.showGrid(x=True, y=True)
        self.rsi_area.setXLink(self.graph_area)

        self.macd_area = pg.PlotWidget()
        self.macd_area.setMaximumHeight(100)
        self.macd_area.setBackground("w")
        self.macd_area.setLabel("left", "MACD", color="k", size="10pt")
        self.macd_area.showGrid(x=True, y=True)
        self.macd_area.setXLink(self.graph_area)

        # Контейнер для панелей
        self.indicator_layout = QtWidgets.QVBoxLayout()
        self.indicator_layout.setSpacing(5)

        # # главный layout
        # graph_layout = QtWidgets.QVBoxLayout()
        # graph_layout.addWidget(self.graph_area)
        # graph_layout.addWidget(self.volume_area)
        # graph_layout.addWidget(self.rsi_area)
        # graph_layout.addWidget(self.macd_area)
        #
        # content_layout.addLayout(graph_layout, 3)

        # главный layout
        graph_layout = QtWidgets.QVBoxLayout()
        graph_layout.addWidget(self.graph_area)
        graph_layout.addWidget(self.volume_area)
        graph_layout.addLayout(self.indicator_layout)  # Динамические панели

        content_layout.addLayout(graph_layout, 3)

        # правый столбец
        right_panel = QtWidgets.QVBoxLayout()

        self.indicator_section = QtWidgets.QGroupBox("Индикаторы")
        indicator_layout = QtWidgets.QVBoxLayout()
        self.indicator_combo = QtWidgets.QComboBox()
        self.indicator_button = QtWidgets.QPushButton("Вывести индикатор")
        self.indicator_button.clicked.connect(self.draw_indicator)
        self.indicator_clear_button = QtWidgets.QPushButton("Очистить индикаторы")
        self.indicator_clear_button.clicked.connect(self.clear_indicators)
        indicator_layout.addWidget(self.indicator_combo)
        indicator_layout.addWidget(self.indicator_button)
        indicator_layout.addWidget(self.indicator_clear_button)
        self.indicator_section.setLayout(indicator_layout)

        self.model_section = QtWidgets.QGroupBox("Модели прогнозирования")
        model_layout = QtWidgets.QVBoxLayout()
        self.model_combo = QtWidgets.QComboBox()
        self.model_button = QtWidgets.QPushButton("Отобразить прогноз")
        self.model_button.clicked.connect(self.draw_forecast)
        self.model_clear_button = QtWidgets.QPushButton("Очистить прогнозы")
        self.model_clear_button.clicked.connect(self.clear_forecasts)
        model_layout.addWidget(self.model_combo)
        model_layout.addWidget(self.model_button)
        model_layout.addWidget(self.model_clear_button)
        self.model_section.setLayout(model_layout)

        # Панель используемых инструментов
        self.active_tools_group = QtWidgets.QGroupBox("Используемые инструменты")
        self.active_tools_group.setVisible(True)
        self.active_tools_group.setMinimumHeight(100)
        # self.active_indicators_group.setMaximumHeight(150)
        self.active_tools_layout = QtWidgets.QVBoxLayout()
        self.active_tools_group.setLayout(self.active_tools_layout)

        right_panel.addWidget(self.indicator_section)
        right_panel.addSpacing(20)
        right_panel.addWidget(self.model_section)
        right_panel.addSpacing(20)
        right_panel.addWidget(self.active_tools_group)
        right_panel.addStretch()

        content_layout.addLayout(right_panel, 1)

        main_layout.addLayout(header_layout)
        main_layout.addSpacing(10)
        main_layout.addLayout(search_layout)
        main_layout.addSpacing(20)
        main_layout.addLayout(content_layout)

        self.load_dynamic_combo_boxes()

    def toggle_chart_mode(self):
        if self.chart_mode == "line":
            self.chart_mode = "candlestick"
        else:
            self.chart_mode = "line"
        self.draw_graph()

    def open_login_window(self):
        from client.ui.login_window import LoginWindow

        self.close()
        self.login_window = LoginWindow()
        self.login_window.show()

    def open_profile_window(self):
        from client.ui.profile_window import ProfileWindow

        self.close()
        self.profile_window = ProfileWindow()
        self.profile_window.show()

    def toggle_date_inputs(self, state):
        enabled = state == QtCore.Qt.Checked
        self.start_date_edit.setEnabled(enabled)
        self.end_date_edit.setEnabled(enabled)

    def get_data(self):
        symbol = self.search_input.text().strip()
        if not symbol:
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Введите тикер актива")
            return

        params = {}
        if self.period_checkbox.isChecked():
            start_date = self.start_date_edit.date().toString("yyyy-MM-dd")
            end_date = self.end_date_edit.date().toString("yyyy-MM-dd")
            params["start"] = start_date
            params["end"] = end_date

        data = get_asset_history(symbol, params=params)

        # data = get_asset_history(symbol)
        if not data:
            QtWidgets.QMessageBox.warning(self, "Ошибка", f"Нет данных для {symbol}")
            return

        self.last_graph_data = data
        self.current_symbol = symbol

        # Очистим все графики
        self.rsi_area.clear()
        self.macd_area.clear()

        self.draw_graph()

    def draw_graph(self):
        self.clear_active_tools()

        # график цен
        self.graph_area.clear()

        data = self.last_graph_data
        symbol = self.current_symbol

        x = [datetime.fromisoformat(point["timestamp"]).timestamp() for point in data]
        if self.chart_mode == "line":
            y = [point["close"] for point in data]
            self.last_graph_data_xy = {"x": x, "y": y}
            self.graph_area.plot(x, y, pen=pg.mkPen(color="b", width=2))
        elif self.chart_mode == "candlestick":
            self.draw_candlestick_chart(data)

        self.graph_area.setTitle(f"График {symbol.upper()}", color="k", size="16pt")

        # график объемов
        self.volume_area.clear()
        # volumes = [point["volume"] for point in data]
        scale_factor = 5  # подобрать вручную от 2 до 10
        volumes = [point["volume"] * scale_factor for point in data]
        bar_items = []

        for i in range(len(x)):
            bar_items.append({
                'x': x[i],
                'height': volumes[i],
                'width': (x[1] - x[0]) * 0.2,  # ширина столбца
                'brush': pg.mkBrush("#808080")
            })

        volume_bar = pg.BarGraphItem(**{
            'x': [item['x'] for item in bar_items],
            'height': [item['height'] for item in bar_items],
            'width': bar_items[0]['width'],
            'brush': bar_items[0]['brush']
        })

        self.volume_area.addItem(volume_bar)

    def draw_candlestick_chart(self, data):
        for point in data:
            timestamp = datetime.fromisoformat(point["timestamp"]).timestamp()
            open_price = point["open"]
            high = point["high"]
            low = point["low"]
            close = point["close"]

            color = 'g' if close >= open_price else 'r'
            pen_thin = pg.mkPen(color=color, width=2)
            pen_thick = pg.mkPen(color=color, width=4)

            self.graph_area.plot([timestamp, timestamp], [low, high], pen=pen_thin)
            self.graph_area.plot([timestamp, timestamp], [open_price, close], pen=pen_thick)

    def predict_arima(self, y_values, last_x):
        if len(y_values) < 30:
            QtWidgets.QMessageBox.warning(self, "Недостаточно данных", "Для прогноза необходимо минимум 30 точек.")
            return [], []

        try:
            model = ARIMA(y_values, order=(2, 1, 2))
            # model = ARIMA(y_values, order=(3, 1, 0))
            # model = ARIMA(y_values, order=(1, 1, 1))
            model_fit = model.fit()

            forecast = model_fit.forecast(steps=5)
            last_date = last_x[-1]

            interval_sec = last_x[-1] - last_x[-2]
            x_pred = [last_date + (i + 1) * interval_sec for i in range(5)]

            return x_pred, forecast.tolist()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка ARIMA", str(e))
            return [], []

    # ------------------------------------------------------------------------------------------------------------------
    # LSTM TORCH v2.0
    # ------------------------------------------------------------------------------------------------------------------
    def predict_lstm(self, y_values, x_values, forecast_steps=5):
        if len(y_values) < 90:
            QtWidgets.QMessageBox.warning(self, "Недостаточно данных", "Для LSTM необходимо минимум 90 точек ({len(y_values)}).")
            return [], []

        class LSTMModel(nn.Module):
            def __init__(self, input_size=1, hidden_size=64, num_layers=2):
                super(LSTMModel, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x, hidden):
                out, hidden = self.lstm(x, hidden)
                out = self.fc(out[:, -1, :])
                return out, hidden

            def init_hidden(self, batch_size):
                return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                        torch.zeros(self.num_layers, batch_size, self.hidden_size))

        # Нормализация данных
        scaler = MinMaxScaler()
        y_scaled = scaler.fit_transform(np.array(y_values).reshape(-1, 1))

        # Формирование обучающей выборки
        x_train = np.array([y_scaled[i - 60:i, 0] for i in range(60, len(y_scaled))])
        y_train = np.array([y_scaled[i, 0] for i in range(60, len(y_scaled))])

        x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(-1)
        y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)

        # Инициализация модели
        model = LSTMModel()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Обучение модели
        model.train()
        for epoch in range(150):
            hidden = model.init_hidden(x_train.size(0))
            output, _ = model(x_train, hidden)
            loss = criterion(output, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Прогнозирование
        model.eval()
        input_seq = torch.tensor(y_scaled[-60:].reshape(1, 60, 1), dtype=torch.float32)
        hidden = model.init_hidden(1)

        predictions = []
        for _ in range(forecast_steps):
            with torch.no_grad():
                next_val, hidden = model(input_seq, hidden)
                next_val_scalar = next_val.item()
                predictions.append(next_val_scalar)
                next_tensor = torch.tensor([[[next_val_scalar]]], dtype=torch.float32)
                input_seq = torch.cat((input_seq[:, 1:, :], next_tensor), dim=1)

        # Обратное преобразование данных
        predictions_rescaled = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

        # Формирование временных меток
        interval = x_values[-1] - x_values[-2]
        x_pred = [x_values[-1] + (i + 1) * interval for i in range(forecast_steps)]

        return x_pred, predictions_rescaled.tolist()
    # ------------------------------------------------------------------------------------------------------------------

    def predict_rnn(self, y_values, x_values, forecast_steps=5):
        window_size = 90
        if len(y_values) < window_size:
            QtWidgets.QMessageBox.warning(
                self, "Недостаточно данных",
                f"Для RNN необходимо минимум {window_size} точек ({len(y_values)})."
            )
            return [], []

        # Определяем модель
        class RNNModel(nn.Module):
            def __init__(self, input_size=1, hidden_size=64, num_layers=2):
                super(RNNModel, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.rnn = nn.RNN(input_size, hidden_size, num_layers,
                                  batch_first=True, nonlinearity='tanh', dropout=0.2)
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x, hidden):
                out, hidden = self.rnn(x, hidden)
                out = self.fc(out[:, -1, :])
                return out, hidden

            def init_hidden(self, batch_size):
                # RNN не имеет состояния ячеек, только hidden
                return torch.zeros(self.num_layers, batch_size, self.hidden_size)

        # 1) нормализация
        scaler = MinMaxScaler()
        y_scaled = scaler.fit_transform(np.array(y_values).reshape(-1, 1))

        # 2) готовим выборку: каждый sample — окно из 60 точек
        x = []
        y = []
        for i in range(window_size, len(y_scaled)):
            x.append(y_scaled[i - window_size:i, 0])
            y.append(y_scaled[i, 0])
        x = np.array(x)  # (N, 60)
        y = np.array(y)  # (N,)

        x_train = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)  # (N, 60, 1)
        y_train = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # (N, 1)

        # 3) инициализируем модель, функцию потерь и оптимизатор
        model = RNNModel()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # 4) обучение
        model.train()
        for epoch in range(120):
            hidden = model.init_hidden(x_train.size(0))
            output, _ = model(x_train, hidden)
            loss = criterion(output, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 5) прогноз
        model.eval()
        seq = torch.tensor(y_scaled[-window_size:].reshape(1, window_size, 1), dtype=torch.float32)
        hidden = model.init_hidden(1)

        preds = []
        for _ in range(forecast_steps):
            with torch.no_grad():
                out, hidden = model(seq, hidden)
                val = out.item()
                preds.append(val)
                new_input = torch.tensor([[[val]]], dtype=torch.float32)
                seq = torch.cat((seq[:, 1:, :], new_input), dim=1)

        # 6) обратное преобразование
        preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

        # 7) генерируем таймстемпы
        interval = x_values[-1] - x_values[-2]
        x_pred = [x_values[-1] + (i + 1) * interval for i in range(forecast_steps)]

        return x_pred, preds.tolist()

    def draw_forecast(self):
        if not hasattr(self, "last_graph_data") or not self.last_graph_data\
                or not hasattr(self, "last_graph_data_xy") or not self.last_graph_data_xy:
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Сначала отобразите график.")
            return

        if self.chart_mode == "candlestick" and self.period_checkbox.isChecked():
            QtWidgets.QMessageBox.information(self, "Прогноз недоступен", "Прогноз работает только для линейного графика.")
            return

        selected_model = self.model_combo.currentText()
        x = self.last_graph_data_xy["x"]
        y = self.last_graph_data_xy["y"]

        try:
            forecast_steps = 5
            x_pred, y_pred = [], []

            if selected_model in self.active_forecasts:
                return

            if selected_model == "ARIMA":
                x_pred, y_pred = self.predict_arima(y, x)
                color = "g"
            elif selected_model == "LSTM":
                # x_pred, y_pred = self.predict_lstm(y, x)
                x_pred, y_pred = self.predict_lstm(y, x, forecast_steps)
                color = "r"
            elif selected_model == "RNN":
                x_pred, y_pred = self.predict_rnn(y, x, forecast_steps)
                color = "m"
            else:
                QtWidgets.QMessageBox.information(self, "Модель", "Метод прогнозирования пока не реализован.")
                return

            if not x_pred or not y_pred:
                QtWidgets.QMessageBox.warning(self, "Модель", "Не удалось построить прогноз.")
                return

            plot_item = self.graph_area.plot(
                x_pred,
                y_pred,
                pen=pg.mkPen(color=color, style=QtCore.Qt.DashLine, width=2),
                name="Прогноз ({selected_model})"
            )

            self.update_active_forecasts(selected_model)
            self.active_forecasts[selected_model] = plot_item

            if self.period_checkbox.isChecked():
                end_date = self.end_date_edit.date().toPyDate()
                # только если конец периода ≤ сегодня−forecast_steps:
                if (datetime.now().date() - end_date).days >= forecast_steps:
                    # 1) тянем ВСЕ торговые точки от end_date+1 до "сейчас"
                    start_str = self.end_date_edit.date().toString("yyyy-MM-dd")
                    end_str = QDate.currentDate().toString("yyyy-MM-dd")
                    real = get_asset_history(self.current_symbol, params={"start": start_str, "end": end_str})

                    # 2) берём ровно первые forecast_steps точек (API вернёт только торговые дни)
                    real = real[1:forecast_steps]
                    x_real = [datetime.fromisoformat(p["timestamp"]).timestamp() for p in real]
                    y_real = [p["close"] for p in real]

                    if not real:
                        QtWidgets.QMessageBox.warning(
                            self, "Недостаточно данных",
                            "Не удалось получить ни одной точки реальных данных для оценки."
                        )
                    else:
                        # 3) для расчёта метрик режем прогноз до длины реальных точек
                        n = len(real)
                        y_pred_cut = y_pred[:n]
                        x_pred_cut = x_pred[:n]

                        # Рисуем реальные
                        self.graph_area.plot(
                            x_real, y_real,
                            pen=pg.mkPen(color="k", style=QtCore.Qt.DotLine, width=2),
                            name="Реальные"
                        )

                        # 4) считаем метрики
                        mae = mean_absolute_error(y_real, y_pred_cut)
                        mse = mean_squared_error(y_real, y_pred_cut)
                        rmse = np.sqrt(mse)
                        mape = float(np.mean(
                            np.abs((np.array(y_real) - np.array(y_pred_cut)) / np.array(y_real))
                        ) * 100)

                        # 5) выводим на график
                        text = (f"MAE: {mae:.4f}\n"
                                f"RMSE: {rmse:.4f}\n"
                                f"MAPE: {mape:.2f}%")
                        ti = pg.TextItem(text, anchor=(0, 1))
                        ti.setPos(x_pred_cut[0], max(y_real + y_pred_cut))
                        self.graph_area.addItem(ti)

                        if n < forecast_steps:
                            QtWidgets.QMessageBox.information(
                                self, "Внимание",
                                f"Получено только {n} реальных точек (из запрошенных {forecast_steps})."
                            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка прогнозирования", f"Ошибка при выполнении прогноза: {str(e)}")

            # if selected_model == "ARIMA":
            #     self.update_active_forecasts("ARIMA")
            #     self.active_indicators["ARIMA"] = plot_item
            # elif selected_model == "LSTM":
            #     self.update_active_forecasts("LSTM")
            #     self.active_indicators["LSTM"] = plot_item
            # elif selected_model == "ARIMA":
            #     self.update_active_forecasts("RNN")
            #     self.active_indicators["RNN"] = plot_item

    def draw_indicator(self):
        if not hasattr(self, "last_graph_data") or not self.last_graph_data \
                or not hasattr(self, "last_graph_data_xy") or not self.last_graph_data_xy:
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Сначала отобразите график.")
            return

        if self.active_indicators.__len__() == 5:
            QtWidgets.QMessageBox.information(self, "Предупреждение", "Максимальное число подключенных"
                                                                      "инструментов - 5.")
            return

        indicator = self.indicator_combo.currentText().strip().upper()

        if indicator in self.active_indicators:
            return

        # Преобразуем данные в DataFrame
        df = pd.DataFrame({
            "timestamp": [datetime.fromtimestamp(ts) for ts in self.last_graph_data_xy["x"]],
            "close": self.last_graph_data_xy["y"]
        }).set_index("timestamp")

        try:
            # SMA
            if indicator == "SMA":
                sma = calculate_sma(df)
                plot_item = self.graph_area.plot(
                    [ts.timestamp() for ts in sma.dropna().index],
                    sma.dropna().values,
                    pen=pg.mkPen("r", width=2),
                    name="SMA"
                )

                self.update_active_indicators("SMA")
                self.active_indicators["SMA"] = plot_item

            # EMA
            elif indicator == "EMA":
                ema = calculate_ema(df)
                plot_item = self.graph_area.plot(
                    [ts.timestamp() for ts in ema.dropna().index],
                    ema.dropna().values,
                    pen=pg.mkPen("m", width=2),
                    name="EMA"
                )

                self.update_active_indicators("EMA")
                self.active_indicators["EMA"] = plot_item

            # RSI
            elif indicator == "RSI":
                self.add_indicator_panel("RSI")
                panel = self.indicator_panels["RSI"]
                panel.clear()

                rsi = calculate_rsi(df)
                panel.plot(
                    [ts.timestamp() for ts in rsi.dropna().index],
                    rsi.dropna().values,
                    pen=pg.mkPen("g", width=2)
                )

                self.update_active_indicators("RSI")
                self.active_indicators["RSI"] = panel

            # MACD
            elif indicator == "MACD":
                self.add_indicator_panel("MACD")
                panel = self.indicator_panels["MACD"]
                panel.clear()

                macd, signal = calculate_macd(df)
                panel.plot(
                    [ts.timestamp() for ts in macd.dropna().index],
                    macd.dropna().values,
                    pen=pg.mkPen("c", width=2)
                )
                panel.plot(
                    [ts.timestamp() for ts in signal.dropna().index],
                    signal.dropna().values,
                    pen=pg.mkPen("y", width=2)
                )

                self.update_active_indicators("MACD")
                self.active_indicators["MACD"] = panel

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Ошибка индикатора", str(e))

    def load_dynamic_combo_boxes(self):
        methods = get_forecasting_methods()
        if methods:
            self.model_combo.clear()
            self.model_combo.addItems(methods)
            self.model_section.show()
        else:
            self.model_section.hide()

        indicators = get_indicator_names()
        if indicators:
            self.indicator_combo.clear()
            self.indicator_combo.addItems(indicators)
            self.indicator_section.show()
        else:
            self.indicator_section.hide()

    def add_indicator_panel(self, name: str):
        if name in self.indicator_panels:
            return

        panel = pg.PlotWidget()
        panel.setBackground("w")
        panel.setMaximumHeight(150)
        panel.setLabel("left", name, color="k", size="10pt")
        panel.showGrid(x=True, y=True)
        panel.setXLink(self.graph_area)

        self.indicator_layout.addWidget(panel)
        self.indicator_panels[name] = panel

    def remove_indicator_panel(self, name: str):
        if name in self.indicator_panels:
            panel = self.indicator_panels.pop(name)
            active_panel = self.active_indicators.pop(name)
            self.indicator_layout.removeWidget(panel)
            panel.deleteLater()

    def remove_indicator(self, name):
        if name in self.indicator_panels:
            self.remove_indicator_panel(name)
        elif name in self.active_indicators:
            item = self.active_indicators.pop(name)
            self.graph_area.removeItem(item)

        if name in self.indicators_buttons:
            button = self.indicators_buttons.pop(name)
            self.active_tools_layout.removeWidget(button)

        self.remove_clear_button()

    def clear_indicators(self):
        for name in list(self.active_indicators.keys()):
            self.remove_indicator(name)

        self.remove_clear_button()

    def update_active_indicators(self, name: str):
        self.add_clear_button()

        if name not in self.active_indicators:
            button = QtWidgets.QPushButton(f"{name} (Удалить)")
            button.setObjectName(name)
            button.clicked.connect(lambda _, n=name: self.remove_indicator(n))
            self.indicators_buttons[name] = button
            self.active_tools_layout.addWidget(button)
            self.active_tools_layout.setSpacing(5)

    def remove_forecast(self, name):
        if name in self.active_forecasts:
            item = self.active_forecasts.pop(name)
            self.graph_area.removeItem(item)

        if name in self.forecasts_buttons:
            button = self.forecasts_buttons.pop(name)
            self.active_tools_layout.removeWidget(button)

        self.remove_clear_button()

    def clear_forecasts(self):
        for name in list(self.active_forecasts.keys()):
            self.remove_forecast(name)

        self.remove_clear_button()

    def update_active_forecasts(self, name: str):
        self.add_clear_button()

        if name not in self.active_indicators:
            button = QtWidgets.QPushButton(f"{name} (Удалить)")
            button.setObjectName(name)
            button.clicked.connect(lambda _, n=name: self.remove_forecast(n))
            self.forecasts_buttons[name] = button
            self.active_tools_layout.addWidget(button)
            self.active_tools_layout.setSpacing(5)

    def add_clear_button(self):
        if self.active_indicators.__len__() == 0 and self.active_forecasts.__len__() == 0:
            self.clear_button = QtWidgets.QPushButton(f"Очистить все инструменты")
            self.clear_button.setObjectName("delete_button")
            self.clear_button.clicked.connect(self.clear_active_tools)
            self.active_tools_layout.addWidget(self.clear_button)
            self.active_tools_layout.setSpacing(15)

    def remove_clear_button(self):
        if self.active_indicators.__len__() == 0 and self.active_forecasts.__len__() == 0:
            # button = self.active_tools_layout.findChild(QPushButton, "delete_button")
            button = self.clear_button
            self.clear_button = None
            self.active_tools_layout.removeWidget(button)

    def clear_active_tools(self):
        self.clear_indicators()
        self.clear_forecasts()
