import os
import json
import requests
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas_market_calendars as mcal

from dateutil.tz import tzlocal
from alpaca_trade_api.rest import TimeFrame


def fetch_historical_data_v2(
    symbols, start_date, end_date, timeframe=TimeFrame.Day, limit=10000
):
    """Fetch historical data using Alpaca's multi-bar API v2 and handle pagination."""

    # Join symbols into a comma-separated string
    symbol_str = ",".join(symbols)

    # Set the base URL for the Alpaca API
    base_url = "https://data.alpaca.markets/v2"

    # Initialize an empty DataFrame to store the results
    data = pd.DataFrame()

    # Initialize the page_token
    page_token = None

    while True:
        # Build the query parameters
        params = {
            "start": start_date,
            "end": end_date,
            "timeframe": timeframe,
            "limit": limit,
            "adjustment": "split",
        }
        if page_token is not None:
            params["page_token"] = page_token

        # Send the GET request to the Alpaca API
        url = f"{base_url}/stocks/bars?symbols={symbol_str}"
        headers = {
            "APCA-API-KEY-ID": os.getenv("APCA_API_KEY_ID"),
            "APCA-API-SECRET-KEY": os.getenv("APCA_API_SECRET_KEY"),
        }
        response = requests.get(url, headers=headers, params=params)

        # Raise an exception if the request was unsuccessful
        response.raise_for_status()

        # Load the response data into a JSON object
        response_json = response.json()

        # Append the data for each symbol to the DataFrame
        for symbol, bars in response_json["bars"].items():
            df = pd.DataFrame(bars)
            df["symbol"] = symbol
            df["t"] = pd.to_datetime(df["t"]).dt.tz_convert(
                tzlocal()
            )  # Convert 't' to datetime and localize
            data = data.append(df)

        # If there's a next_page_token, update the page_token and continue the loop
        page_token = response_json.get("next_page_token")
        if page_token is None:
            break

    return data


def calculate_start_date(ndays):
    """Calculate the start date given the number of trading days from today."""
    nyse = mcal.get_calendar("NYSE")
    end_date = pd.Timestamp.now().normalize()
    start_date = pd.Timestamp("2000-01-01")
    trading_days = nyse.valid_days(start_date=start_date, end_date=end_date)

    if len(trading_days) < ndays:
        raise ValueError(
            "The number of trading days requested is more than the available trading days."
        )

    start_date = trading_days[-ndays]

    return start_date


def compute_correlation_matrix(df, column):
    # Set 't' as the index
    df.set_index("t", inplace=True)

    # Pivot the DataFrame so each symbol is a column
    df_pivot = df.pivot(columns="symbol", values=column)

    # Compute z-scores for each symbol to normalize
    df_zscore = (df_pivot - df_pivot.mean()) / df_pivot.std()

    # Compute the correlation matrix
    correlation_matrix = df_zscore.corr()

    return correlation_matrix


def sort_correlation_matrix(correlation_matrix, sort_by="mean"):
    """Sorts the correlation matrix based on a specified metric (max, min, mean, median, or sum)."""
    metrics = {
        "max": correlation_matrix.max(),
        "min": correlation_matrix.min(),
        "mean": correlation_matrix.mean(),
        "median": correlation_matrix.median(),
        "sum": correlation_matrix.sum(),
    }

    metric = metrics.get(sort_by)

    if metric is None:
        raise ValueError(
            "Invalid sort_by value. Must be one of 'max', 'min', 'mean', 'median', or 'sum'."
        )

    # Sort the columns based on the chosen metric
    sorted_columns = metric.sort_values(ascending=False).index

    # Reindex the DataFrame to sort it
    sorted_correlation_matrix = correlation_matrix.reindex(
        index=sorted_columns, columns=sorted_columns
    )

    return sorted_correlation_matrix, metric


def plot_correlation_matrix(correlation_matrix, colormap, title="Correlation Matrix"):
    """Plot the correlation matrix."""

    f = plt.figure(figsize=(12, 13))
    plt.matshow(
        correlation_matrix, fignum=f.number, cmap=plt.get_cmap(colormap)
    )
    plt.xticks(
        range(correlation_matrix.shape[1]),
        correlation_matrix.columns,
        fontsize=7,
        rotation=45,
    )
    plt.yticks(
        range(correlation_matrix.shape[1]),
        correlation_matrix.columns,
        fontsize=7,
    )
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=7)
    plt.title(title, fontsize=12)
    plt.show(block=False)


def plot_time_series(data, sorted_symbols, sort_by, metric, column):
    """Plot the time series for each symbol."""

    # Pivot the DataFrame so each symbol is a column
    data_pivot = data.pivot(columns="symbol", values=column)

    # Reorder the columns based on the sorted symbols
    data_pivot = data_pivot[sorted_symbols]

    # Create lineplots
    list_length = data_pivot.shape[1]
    ncols = 6
    nrows = int(np.ceil(list_length / ncols))
    height = list_length / 3 if list_length > 30 else 14

    fig, axs = plt.subplots(
        nrows=nrows, ncols=ncols, sharex=True, figsize=(14, height)
    )

    # Add a title to the figure
    fig.text(
        0.5,
        0.98,
        f"Cross-correlation Time-series Sorted by {sort_by} (Based on {column} column)",
        ha="center",
        va="top",
        fontsize=14,
    )

    for i, ax in enumerate(axs.flatten()):
        if i < list_length:
            sns.lineplot(
                data=data_pivot,
                x=data_pivot.index,
                y=data_pivot.iloc[:, i],
                ax=ax,
                color="dodgerblue",
                linewidth=0.7,
            )

            # Add sort value to the title
            sort_value = round(metric[data_pivot.columns[i]], 3)
            ax.set_title(
                f"{data_pivot.columns[i]} ({sort_by}: {sort_value})",
                fontsize=8,
            )

            ax.grid(linestyle="--", linewidth=0.40)
            ax.tick_params(labelrotation=45, labelsize=6)
        else:
            ax.axis("off")  # Hide unused axes

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def run(args):
    """Main function to control the flow of the program."""

    # Read the list of symbols from the file
    with open(args.list, "r") as f:
        symbols = [line.strip().upper() for line in f]

    # Calculate the start date
    start_date = calculate_start_date(args.ndays).strftime("%Y-%m-%d")
    end_date = pd.Timestamp.now().tz_localize(tzlocal()).strftime("%Y-%m-%d")

    # Fetch the historical data
    data = fetch_historical_data_v2(symbols, start_date, end_date)

    print(data)

    # Compute the correlation matrix
    correlation_matrix = compute_correlation_matrix(data, args.column)

    # Sort the correlation matrix
    sorted_correlation_matrix, metric = sort_correlation_matrix(
        correlation_matrix, sort_by=arguments.sort
    )

    # Plot the sorted correlation matrix
    plot_correlation_matrix(
        sorted_correlation_matrix,
        colormap=args.colormap,
        title=f"Stocks Correlation Matrix (Based on {args.column} column)",
    )

    # Plot time series of each stock
    plot_time_series(
        data,
        sorted_symbols=sorted_correlation_matrix.columns,
        sort_by=args.sort,
        metric=metric,
        column=args.column,
    )


if __name__ == "__main__":

    colormap_choices = [
        "viridis", "plasma", "inferno", "magma", "cividis",
        "Greys", "Purples", "Blues", "Greens", "Oranges", "Reds",
        "YlOrBr", "YlOrRd", "OrRd", "PuRd", "RdPu", "BuPu",
        "GnBu", "PuBu", "YlGnBu", "PuBuGn", "BuGn", "YlGn",
        "binary", "gist_yarg", "gist_gray", "gray", "bone", "pink",
        "spring", "summer", "autumn", "winter", "cool", "Wistia",
        "hot", "afmhot", "gist_heat", "copper",
        "PiYG", "PRGn", "BrBG", "PuOr", "RdGy", "RdBu", "RdBu_r",
        "RdYlBu", "RdYlGn", "Spectral", "coolwarm", "bwr", "seismic",
        "twilight", "twilight_shifted", "hsv",
        "Pastel1", "Pastel2", "Paired", "Accent",
        "Dark2", "Set1", "Set2", "Set3",
        "tab10", "tab20", "tab20b", "tab20c",
        "flag", "prism", "ocean", "gist_earth", "terrain", "gist_stern",
        "gnuplot", "gnuplot2", "CMRmap", "cubehelix", "brg",
        "gist_rainbow", "rainbow", "jet", "nipy_spectral", "gist_ncar"
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--list",
        "-l",
        type=str,
        required=True,
        help="File containing list of stock symbols",
    )
    parser.add_argument(
        "--ndays",
        "-n",
        type=int,
        default=504,
        help="Number of days to retrieve historical prices (default=504)",
    )
    parser.add_argument(
        "--sort",
        "-s",
        type=str,
        default="mean",
        choices=["max", "min", "mean", "median", "sum"],
        help="Method for sorting the correlation matrix (default=mean)",
    )
    parser.add_argument(
        "--column",
        "-c",
        type=str,
        default="c",
        choices=["o", "h", "l", "c", "v", "n", "vw"],
        help="Column to be used for the cross-correlation calculation (default='c')",
    )
    parser.add_argument(
        "--colormap",
        "-cm",
        type=str,
        default="RdBu_r",
        choices=colormap_choices,
        help="Colormap to be used for the correlation matrix plot (default='RdBu_r')",
    )
    arguments = parser.parse_args()

    run(arguments)
