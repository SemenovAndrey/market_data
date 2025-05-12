import pandas as pd

# def calculate_sma(data: pd.DataFrame, period: int = 14) -> pd.Series:
#     return data["close"].rolling(window=period).mean()
#
# def calculate_ema(data: pd.DataFrame, span: int = 14) -> pd.Series:
#     return data["close"].ewm(span=span, adjust=False).mean()
#
# def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
#     delta = data["close"].diff()
#     gain = delta.where(delta > 0, 0)
#     loss = -delta.where(delta < 0, 0)
#
#     avg_gain = gain.rolling(window=period).mean()
#     avg_loss = loss.rolling(window=period).mean()
#
#     rs = avg_gain / avg_loss
#     return 100 - (100 / (1 + rs))
#
# def calculate_macd(data: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
#     ema12 = data["close"].ewm(span=12, adjust=False).mean()
#     ema26 = data["close"].ewm(span=26, adjust=False).mean()
#     macd_line = ema12 - ema26
#     signal_line = macd_line.ewm(span=9, adjust=False).mean()
#     return macd_line, signal_line


def log_data_stats(data: pd.Series, label: str):
    print(f"\n{label} - Статистика данных:")
    print(f"  Min: {data.min()}")
    print(f"  Max: {data.max()}")
    print(f"  Среднее: {data.mean()}")
    print(f"  Std: {data.std()}")
    print(f"  Кол-во NaN: {data.isna().sum()}")


def calculate_sma(data: pd.DataFrame, period: int = 14) -> pd.Series:
    # if "close" not in data:
    #     raise ValueError("В данных отсутствует колонка 'close'")

    sma = data["close"].rolling(window=period).mean()
    log_data_stats(sma, "SMA")
    return sma


def calculate_ema(data: pd.DataFrame, span: int = 14) -> pd.Series:
    # if "close" not in data:
    #     raise ValueError("В данных отсутствует колонка 'close'")

    ema = data["close"].ewm(span=span, adjust=False).mean()
    log_data_stats(ema, "EMA")
    return ema


def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
    # if "close" not in data:
    #     raise ValueError("В данных отсутствует колонка 'close'")

    delta = data["close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    log_data_stats(rsi, "RSI")

    # Проверка на значения RSI
    if rsi.min() < 0 or rsi.max() > 100:
        print("⚠️  ВНИМАНИЕ: Значения RSI вышли за допустимые границы (0-100).")

    return rsi


def calculate_macd(data: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    # if "close" not in data:
    #     raise ValueError("В данных отсутствует колонка 'close'")

    ema12 = data["close"].ewm(span=12, adjust=False).mean()
    ema26 = data["close"].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()

    log_data_stats(macd_line, "MACD Line")
    log_data_stats(signal_line, "Signal Line")

    # Проверка на малое отклонение MACD
    macd_range = macd_line.max() - macd_line.min()
    if macd_range < 0.1:
        print(f"⚠️  ВНИМАНИЕ: MACD слишком мал. Диапазон: {macd_range}")

    return macd_line, signal_line
