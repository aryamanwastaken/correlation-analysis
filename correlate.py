#! /usr/bin/python3

import os
import argparse
import numpy as np
import concurrent.futures
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas_market_calendars as mcal

from alpaca_trade_api.rest import REST, TimeFrame

API_KEY_ID = os.environ["APCA_API_KEY_ID"]
SECRET_KEY_ID = os.environ["APCA_API_SECRET_KEY"]
BASE_URL = os.environ["APCA_API_BASE_URL"]

rest_api = REST(API_KEY_ID, SECRET_KEY_ID, BASE_URL)

pd.set_option("display.width", 1000)
pd.set_option("display.max_rows", None, "display.max_columns", None)


def _get_alpaca_prices(
    symbols, _from, last_open_date, num_samps, TIMEFRAME, workers=10
):
    """Get the map of DataFrame price data from alpaca, in parallel."""

    def historic_prices_v2(symbol):
        try:
            df = (
                rest_api.get_bars(
                    symbol,
                    TIMEFRAME,
                    _from,
                    last_open_date,
                    adjustment="split",
                )
                .df[["close"]]
                .tail(num_samps)
            )
        except Exception:
            return
        return df.tail(num_samps)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=workers
    ) as executor:
        results = {}
        future_to_symbol = {
            executor.submit(historic_prices_v2, symbol): symbol
            for symbol in symbols
        }
        for future in concurrent.futures.as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                results[symbol] = future.result()
            except Exception as exc:
                print("{} generated an exception: {}".format(symbol, exc))
        return results


def run(args):
    """Fetch historical data, process, and plot results"""

    stock_list = str(args.list)
    sample_interval = "{}".format(str(args.sample))
    ref_symbol = "{}".format(str(args.ref).upper())
    num_samps = int(args.ns)
    num_days = int(args.ndays)

    TIMEFRAME = TimeFrame.Minute
    if sample_interval == "Day":
        TIMEFRAME = TimeFrame.Day
        if num_days < num_samps:
            num_days = int(1.5 * num_samps)
            print('Modifying "ndays" parameter = %d' % num_days)

    tickers = [el.symbol for el in rest_api.list_assets(status="active")]

    nyse = mcal.get_calendar("NYSE")

    end_dt = pd.Timestamp.now(tz="America/New_York")
    start_dt = end_dt - pd.Timedelta("%4d days" % num_days)
    _from = start_dt.strftime("%Y-%m-%d")
    to_end = end_dt.strftime("%Y-%m-%d")

    nyse_schedule = (
        nyse.schedule(start_date=_from, end_date=to_end)
        .tail(1)
        .index.to_list()
    )
    last_open_date = str(nyse_schedule[0]).split(" ")[0]

    with open(stock_list + ".lis", "r") as f:
        universe = [row.split()[0] for row in f]
    f.close()
    universe[:] = [item.upper() for item in universe if item != ""]

    [ticker for ticker in tickers if ticker in universe]
    if ref_symbol not in universe:
        universe.append(ref_symbol)
    if "UVXY" not in universe:
        universe.append("UVXY")

    price_map = _get_alpaca_prices(
        universe, _from, last_open_date, num_samps, TIMEFRAME
    )

    df_ref = pd.DataFrame(index=price_map[ref_symbol].index)
    df_ref = price_map[ref_symbol]
    df_whole = df_ref
    universe.remove(ref_symbol)
    df_whole.rename(columns={"close": ref_symbol}, inplace=True)

    for symbol in universe:
        df1 = pd.DataFrame(index=pd.to_datetime(price_map[symbol].index))
        df1 = pd.DataFrame(price_map[symbol])
        if df1.shape[0] == num_samps:
            df_whole = df_whole.merge(df1, left_index=True, right_index=True)
            df_whole = df_whole.rename(columns={"close": symbol})
    df = df_whole.ffill().bfill().dropna(axis=1, how="all").tail(num_samps)

    hundred = 100.0
    for symbol, prices in df.iteritems():
        first_price = prices[0]
        prices = ((prices - first_price) / first_price) * hundred
        df[symbol] = prices.values

    df_corr = df.corr()
    df_mean = df_corr.mean().sort_values(ascending=False).dropna()
    print(df_mean.sort_values(ascending=True))
    mean_list = df_mean.index.tolist()

    df1 = df.reindex(columns=mean_list)

    f = plt.figure(figsize=(12, 13))
    plt.matshow(df1.corr(), fignum=f.number, cmap=plt.get_cmap("RdBu_r"))
    plt.xticks(
        range(df1.select_dtypes(["number"]).shape[1]),
        df1.select_dtypes(["number"]).columns,
        fontsize=7,
        rotation=45,
    )
    plt.yticks(
        range(df1.select_dtypes(["number"]).shape[1]),
        df1.select_dtypes(["number"]).columns,
        fontsize=7,
    )
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=7)
    plt.title(str(stock_list).upper() + " Correlation Matrix", fontsize=12)
    plt.show()

    print(" ")
    print("Relative correlation with reference = ", ref_symbol)
    print(" ")
    rel_corr = pd.DataFrame(index=universe, columns=["correlation_result"])
    symbol_list = list(df1.columns)
    for symbol in symbol_list:
        result = df1[ref_symbol].corr(df1[symbol])
        rel_corr.loc[symbol] = [result]

    rel_corr_sorted = rel_corr.sort_values(
        by=["correlation_result"], ascending=[1]
    ).dropna()

    pd.options.display.float_format = "{:,.4f}".format
    print(rel_corr_sorted)

    df1.reset_index(inplace=True)
    df1.drop(
        ["timestamp"],
        axis=1,
        inplace=True,
    )
    corr = df1.corr()

    # Create lineplots
    list_length = df1.shape[1]
    ncols = 6
    nrows = int(round(list_length / ncols, 0))
    height = list_length / 3 if list_length > 30 else 14

    fig, axs = plt.subplots(
        nrows=nrows, ncols=ncols, sharex=True, figsize=(14, height)
    )

    for i, ax in enumerate(fig.axes):
        if i < list_length:
            sns.lineplot(data=df1, x=df1.index, y=df1.iloc[:, i], ax=ax)
            ax.set_title(df1.columns[i])
            ax.grid(linestyle="--", linewidth=0.40)
            ax.tick_params(labelrotation=45)

    plt.tight_layout()
    plt.show()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = "RdBu_r"

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
    )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        "--list",
        type=str,
        default="momentum",
        help="List of possible trade Symbols, (no default, ex: momentum)",
    )
    PARSER.add_argument(
        "--ns",
        type=int,
        default=1000,
        help="Number of samples to retrieve historical prices (default=1000)",
    )
    PARSER.add_argument(
        "--ndays",
        type=int,
        default=8,
        help="Number of days to retrieve historical prices (default=8)",
    )
    PARSER.add_argument(
        "--sample",
        type=str,
        default="Minute",
        help="Sample increment - minute or day (Default=Minute)",
    )
    PARSER.add_argument(
        "--ref",
        type=str,
        default="SPY",
        help="Reference symbol for correlation matrix (default=SPY)",
    )
    ARGUMENTS = PARSER.parse_args()

    run(ARGUMENTS)
