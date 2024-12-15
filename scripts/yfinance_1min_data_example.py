
import pandas as pd
import seaborn as sns
import yfinance as yf
import matplotlib.pyplot as plt
import datetime

from src.definitions import SPX_INDEX_DATA, SPX_FUTURE_DATA

sns.set_theme()


def download_minute_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    ticker: yf.Ticker = yf.Ticker(symbol)
    data: pd.DataFrame = ticker.history(
        start=start_date,
        end=end_date,
        interval="1m",
    )
    return data

def show_data_stats(data: pd.DataFrame):
    print(data.head())
    print(data.tail())
    num_rows: int = len(data)
    num_days: int = pd.Series(data.index.date).nunique()
    print(f"Total rows: {num_rows:,}")
    print(f"Days: {num_days}")

def split_date_range(start_date: datetime.date, end_date: datetime.date, chunk_days: int):
    total_days = (end_date - start_date).days
    chunks = []

    for i in range(0, total_days, chunk_days):
        chunk_start = start_date + datetime.timedelta(days=i)
        chunk_end = min(chunk_start + datetime.timedelta(days=chunk_days), end_date)
        chunks.append((chunk_start, chunk_end))

    return chunks

if __name__ == "__main__":
    end_date = datetime.date(2024,12,14)

    # Define the start date for 14 days ago
    start_date = end_date - datetime.timedelta(days=14)

    # Split the 14-day range into chunks of 7 days (or any other chunk size)
    chunk_size = 7
    date_chunks = split_date_range(start_date, end_date, chunk_size)

    # Download and concatenate SPX index data
    spx_index_data = pd.DataFrame()
    for chunk_start, chunk_end in date_chunks:
        data_chunk = download_minute_data(symbol="^GSPC", start_date=chunk_start.strftime('%Y-%m-%d'), end_date=chunk_end.strftime('%Y-%m-%d'))
        spx_index_data = pd.concat([spx_index_data, data_chunk])

    spx_index_data.to_csv(SPX_INDEX_DATA)
    show_data_stats(spx_index_data)

    # Download and concatenate SPX future data
    spx_future_data = pd.DataFrame()
    for chunk_start, chunk_end in date_chunks:
        data_chunk = download_minute_data(symbol="ES=F", start_date=chunk_start.strftime('%Y-%m-%d'), end_date=chunk_end.strftime('%Y-%m-%d'))
        spx_future_data = pd.concat([spx_future_data, data_chunk])

    spx_future_data.to_csv(SPX_FUTURE_DATA)
    show_data_stats(spx_future_data)

    # Plotting the data
    plt.figure(figsize=(12, 6))
    plt.plot(spx_future_data.index, spx_future_data["Close"], label="Close E-Mini")
    plt.plot(spx_index_data.index, spx_index_data["Close"], label="Close SPX")

    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Close Price (USD)", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.show()
