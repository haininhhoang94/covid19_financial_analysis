# %%
# import library
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime

sns.set(style="darkgrid")

from pylab import rcParams

rcParams["figure.figsize"] = 15, 8

from functools import reduce
import operator

import os
import requests

import holoviews as hv
import hvplot.pandas

# %%
while True:
    try:
        nyse_hist = pd.read_csv("./Data/NYSE.csv")
        nyse_hist_10_years = pd.read_csv("./Data/NYSE_10years.csv")
        nyse_hist["Date"] = pd.to_datetime(nyse_hist["Date"])
        nyse_hist_10_years["Date"] = pd.to_datetime(nyse_hist_10_years["Date"])
        nyse_hist.set_index("Date", inplace=True)
        nyse_hist_10_years.set_index("Date", inplace=True)
        break
    except FileNotFoundError:
        nyse = yf.Ticker("^NYA")
        nyse.info
        nyse_hist = nyse.history(period="365d")
        nyse_hist.to_csv("./Data/NYSE.csv")
        nyse_hist_10_years = nyse.history(period="3650d")
        nyse_hist_10_years.to_csv("./Data/NYSE_10years.csv")

# %%
# Test
# get covid-19 info
# https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge/discussion/141607
# https://towardsdatascience.com/covid-19-data-collection-a-python-api-story-347aafa95e69
def covid_19_data_collection(countries=["US", "FR"]):
    import COVID19Py as cpy

    covid = cpy.COVID19(url="https://covid-tracker-us.herokuapp.com")

    for i in range(len(countries)):
        all_location_json = covid.getLocationByCountryCode(
            countries[i], timelines=True
        )[0]

        confirmed = pd.DataFrame.from_dict(
            all_location_json["timelines"]["confirmed"]["timeline"], orient="index",
        ).rename(columns={0: "{}_confirmed".format(countries[i])})

        # deaths = pd.DataFrame.from_dict(
        #    all_location_json["timelines"]["deaths"]["timeline"], orient="index",
        # ).rename(columns={0: "{}_deaths".format(countries[i])})

        # recovered = pd.DataFrame.from_dict(
        #     all_location_json["timelines"]["recovered"]["timeline"], orient="index",
        # ).rename(columns={0: "{}_recovered".format(countries[i])})

        if i == 0:
            covid_df = confirmed
        else:
            covid_df = covid_df.join(confirmed)
        # covid_df = covid_df.join(deaths)
        # covid_df = covid_df.join(recovered)
    return covid_df


# %%

countries = ["US", "FR", "ES", "ZA", "BR", "RU"]
covid_df = covid_19_data_collection(countries)
covid_df.index = pd.to_datetime(covid_df.index).date

# %%
# get unployment rate US information
# https://www.bls.gov/developers/api_python.htm
# Unemployment Rate - LNS14000000
# Discouraged Workers - LNU05026645
# Persons At Work Part Time for Economic Reasons - LNS12032194
# Unemployment Rate - 25 Years & Over, Some College or Associate Degree - LNS14027689
def unemployment_rate(start_year=2019, end_year=2020):
    import requests
    import json

    headers = {"Content-type": "application/json"}
    data = json.dumps(
        {
            "seriesid": ["LNS14000000"],
            "startyear": "{}".format(start_year),
            "endyear": "{}".format(end_year),
        }
    )
    p = requests.post(
        "https://api.bls.gov/publicAPI/v2/timeseries/data/", data=data, headers=headers
    )
    json_data = json.loads(p.text)
    result = pd.DataFrame.from_dict(json_data["Results"]["series"][0]["data"])
    return result


# %%

unemployment_df = unemployment_rate(start_year=2015, end_year=2020).drop(
    columns="footnotes"
)

unemployment_df["Date"] = (
    unemployment_df["year"] + "-" + unemployment_df["periodName"] + "-" + "01"
)
unemployment_df["Date"] = pd.to_datetime(unemployment_df["Date"], format="%Y-%B-%d")
unemployment_df.set_index("Date", inplace=True)

# %%
# Get business cycle:
# Business cycle is the data/graph of movement in GDP around its long-term growth trend
# https://codingandfun.com/economic-indicators-with-python/
def checkindicator(url):
    import requests
    import json

    r = requests.get(url)
    r = r.json()
    periods = r["series"]["docs"][0]["period"]
    values = r["series"]["docs"][0]["value"]
    dataset = r["series"]["docs"][0]["dataset_name"]

    indicators = pd.DataFrame(values, index=periods)
    indicators.columns = [dataset]
    return indicators


# %%

GDPgrowth = checkindicator(
    "https://api.db.nomics.world/v22/series/WB/WDI/NY.GDP.MKTP.KD.ZG-EU?observations=1"
)

nyse_hist_10_years = nyse_hist

# %%
# # Visualize the data
# NYSE price from 2019 to 2020

from pylab import rcParams

rcParams["figure.figsize"] = 15, 8
# plt.figure(figsize=(15,8))
nyse_plot = sns.lineplot(x=nyse_hist.index.values, y="Close", data=nyse_hist)


nyse_plot_10_years = sns.lineplot(
    x=nyse_hist_10_years.index.values, y="Close", data=nyse_hist_10_years
)

# %%
from pylab import rcParams

rcParams["figure.figsize"] = 15, 8

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()

ax1.plot(nyse_hist["Close"].loc["2020-02-01":], "--b", label="NYSE")
ax2.plot(nyse_hist["Volume"].loc["2020-02-01":], "-r")

ax1.set_xlabel("Date")
ax1.set_ylabel("NYSE Close Price", color="g")
ax2.set_ylabel("NYSE Volume", color="r")

plt.legend()
plt.show()
# %%

# COVID-19 Infection Rate

# plt.figure(figsize=(15,8))
covid_plot = sns.lineplot(data=covid_df.drop(columns="FR_confirmed"), dashes=False)

# ## Combine graph

nyse_hist_close = nyse_hist.drop(
    columns=["Open", "High", "Low", "Volume", "Dividends", "Stock Splits"]
)
combined_df = nyse_hist_close.join(covid_df)
combined_df.head()

# %%
fig, ax1 = plt.subplots()

ax2 = ax1.twinx()

ax1.plot(combined_df["Close"].loc["2020-02-01":], "--b", label="NYSE")
ax2.plot(combined_df.drop(columns="Close").loc["2020-02-01":])

ax1.set_xlabel("Date")
ax1.set_ylabel("NYSE Close Price", color="g")
ax2.set_ylabel("Confirmed Case", color="r")

plt.legend(combined_df.drop(columns="Close").columns.values)
plt.show()

# %%
# Create plotting object
plot_data = hv.Dataset(nyse_hist, kdims=["Date"], vdims=["Close"])

# %%
# Create scatter plot

black_tuesday = pd.to_datetime("2020-03-15")

vline = hv.VLine(black_tuesday).options(color="#FF7E47")

m = (
    hv.Scatter(plot_data)
    .options(width=700, height=400)
    .redim("NYSE Share Trading Close Price")
    .hist()
    * vline
    * hv.Text(
        black_tuesday + pd.DateOffset(months=10), 4e7, "Covid-19 Crash", halign="left"
    ).options(color="#FF7E47")
)
m
# %%

# ## Chow Test

# +
# https://github.com/jtloong/chow_test/blob/master/tests/test_analysis.ipynb

import numpy as np
from scipy.stats import f


def f_value(y1, x1, y2, x2):
    """This is the f_value function for the Chow Break test package
    Args:
        y1: Array like y-values for data preceeding the breakpoint
        x1: Array like x-values for data preceeding the breakpoint
        y2: Array like y-values for data occuring after the breakpoint
        x2: Array like x-values for data occuring after the breakpoint

    Returns:
        F-value: Float value of chow break test
    """

    def find_rss(y, x):
        """This is the subfunction to find the residual sum of squares for a given set of data
        Args:
            y: Array like y-values for data subset
            x: Array like x-values for data subset

        Returns:
            rss: Returns residual sum of squares of the linear equation represented by that data
            length: The number of n terms that the data represents
        """
        A = np.vstack([x, np.ones(len(x))]).T
        rss = np.linalg.lstsq(A, y, rcond=None)[1]
        length = len(y)
        return (rss, length)

    rss_total, n_total = find_rss(np.append(y1, y2), np.append(x1, x2))
    rss_1, n_1 = find_rss(y1, x1)
    rss_2, n_2 = find_rss(y2, x2)

    chow_nom = (rss_total - (rss_1 + rss_2)) / 2
    chow_denom = (rss_1 + rss_2) / (n_1 + n_2 - 4)
    return chow_nom / chow_denom


# %%
def p_value(y1, x1, y2, x2, **kwargs):
    F = f_value(y1, x1, y2, x2, **kwargs)
    if not F:
        return 1
    df1 = 2
    df2 = len(x1) + len(x2) - 4

    # The survival function (1-cdf) is more precise than using 1-cdf,
    # this helps when p-values are very close to zero.
    # -f.logsf would be another alternative to directly get -log(pval) instead.
    p_val = f.sf(F[0], df1, df2)
    return p_val


# %%

y1 = nyse_hist[nyse_hist.index < pd.to_datetime("2020-01-01")]["Volume"]
x1 = nyse_hist[nyse_hist.index < pd.to_datetime("2020-01-01")]["Close"]
y2 = nyse_hist[nyse_hist.index >= pd.to_datetime("2020-01-01")]["Volume"]
x2 = nyse_hist[nyse_hist.index >= pd.to_datetime("2020-01-01")]["Close"]

f_test = f_value(y1, x1, y2, x2)
f_test

p_val = p_value(y1, x1, y2, x2)
p_val

# ## This p-value < 0.05 indicate that there is a structure break

# %%
# Create plotting object
plot_data = hv.Dataset(nyse_hist, kdims=["Date"], vdims=["Volume"])

# Create scatter plot

black_tuesday = pd.to_datetime("2020-03-15")

vline = hv.VLine(black_tuesday).options(color="#FF7E47")

m = (
    hv.Scatter(plot_data)
    .options(width=700, height=400)
    .redim("NYSE Share Trading Volume")
    .hist()
    * vline
    * hv.Text(
        black_tuesday + pd.DateOffset(months=10), 4e7, "Covid-19 Crash", halign="left"
    ).options(color="#FF7E47")
)
m

# %%
# %%opts Scatter [width=400 height=200]

nyse_hist["Date_"] = nyse_hist.index.values
nyse_hist["Quarter"] = nyse_hist.Date_.dt.quarter

# %%


def second_order(days_window):
    data_imputed = nyse_hist
    data_imputed.Volume = data_imputed.Volume.interpolate()

    return hv.Scatter(
        pd.concat(
            [data_imputed.Date_, data_imputed.Volume.rolling(days_window).mean()],
            names=["Date", "Volumne Trend"],
            axis=1,
        ).dropna()
    ).redim(Volume="Mean Trend") + hv.Scatter(
        pd.concat(
            [data_imputed.Date_, data_imputed.Volume.rolling(days_window).cov()],
            names=["Date", "Volumne Variance"],
            axis=1,
        ).dropna()
    ).redim(
        Volume="Volume Variance"
    ).options(
        color="#FF7E47"
    )


hv.DynamicMap(second_order, kdims=["days_window"]).redim.range(days_window=(7, 1000))

# %%
# ## ACF and PACF Volume

# %%
# # %%opts Bars [width=400 height=300]
from statsmodels.tsa.stattools import acf, pacf


def auto_correlations(start_year, window_years):
    start_year = pd.to_datetime(f"{start_year}-01-01")
    window_years = pd.DateOffset(years=window_years)

    data_window = nyse_hist
    data_window = data_window.loc[
        (
            (data_window.Date_ >= start_year)
            & (data_window.Date_ <= (start_year + window_years))
        ),
        :,
    ]

    return hv.Bars(acf(data_window.Volume.interpolate().dropna())).redim(
        y="Autocorrelation", x="Lags"
    ) + hv.Bars(pacf(data_window.Volume.interpolate().dropna())).redim(
        y="Patial Autocorrelation", x="Lags"
    ).options(
        color="#FF7E47"
    )


hv.DynamicMap(auto_correlations, kdims=["start_year", "window_years"]).redim.range(
    start_year=(nyse_hist.Date_.min().year, nyse_hist.Date_.max().year),
    window_years=(1, 25),
)

# %%
# Clearly see a prediction/panic mode in NYSE before the first case in Covid-19 happened

# %%
unemployment_df.head()
combined_df_2 = combined_df.join(unemployment_df).drop(
    columns=["year", "period", "periodName", "latest"]
)

combined_df_2 = combined_df_2.dropna().rename(columns={"value": "Unployment Rate"})
combined_df_2

# Graph a chart plot to analyze the unemployment rate

# %%
fig, ax1 = plt.subplots()

ax2 = ax1.twinx()

ax1.bar(combined_df_2.index.values, combined_df_2["Unployment Rate"])
ax2.plot(
    combined_df_2.index.values, combined_df_2.drop(columns=["Close", "Unployment Rate"])
)

ax1.set_xlabel("Date")
ax1.set_ylabel("NYSE Close Price", color="g")
ax2.set_ylabel("Confirmed Case", color="r")

# plt.legend(combined_df.drop(columns='Close').columns.values)
plt.show()
# %%

# NYSE predict the unemployment and GDP

# ## ACF and PACF for Price
# Inspect to see pattern in NYSE. Since ACF and PACF is to analyze seasonal/pattern, we will apply the technique to NYSE 10 years

# %%
import statsmodels.api as sm

sm.graphics.tsa.plot_acf(nyse_hist_10_years["Close"])
plt.show()

# %%
sm.graphics.tsa.plot_pacf(nyse_hist_10_years["Close"])
plt.show()

# %%

# With log return

# %%
def nans(shape, dtype=float):
    # To generate nans array
    a = np.empty(shape, dtype)
    a.fill(np.nan)
    return a


def shift(price_array, n):
    # n is the number shift to the right
    result = nans(len(price_array))
    for i in range(n, len(price_array)):
        result[i] = price_array[i - n]
    return result


def Normal_Return(P_f, P_i):
    result = (P_f - P_i) / P_i
    return result


def Log_Return(P_f, P_i):
    result = np.log(P_f / P_i)
    return result


# %%

nyse_hist_10_years["Log Return"] = Log_Return(
    nyse_hist_10_years["Close"], shift(nyse_hist_10_years["Close"], 1)
)

nyse_hist_10_years.head()

sm.graphics.tsa.plot_acf(nyse_hist_10_years["Log Return"])
plt.show()

sm.graphics.tsa.plot_pacf(nyse_hist_10_years["Log Return"])
plt.show()

# Cannot use ACF and PACF with log return

# ## Random Walk

# %%
from functools import reduce
import operator

import os
import requests

import pandas as pd
import numpy as np

import holoviews as hv
import hvplot.pandas

# %%

np.random.seed(42)
hv.extension("bokeh")
# --

# %%
def plot(mu, sigma, samples):
    return (
        pd.Series(np.random.normal(mu, sigma, 1000))
        .cumsum()
        .hvplot(title="Random Walks", label=f"{samples}")
    )


def prod(mu, sigma, samples):
    return reduce(
        operator.mul, list(map(lambda x: plot(mu, sigma, x), range(1, samples + 1)))
    )


# %%
hv.DynamicMap(prod, kdims=["mu", "sigma", "samples"]).redim.range(
    mu=(14000, 14001), sigma=(100000, 100001), samples=(30, 50)
).options(width=900, height=400)

# Random Walk for NYSE from today

# ## AS-AD Model

# Using economic growth

# %%
from scipy.optimize import fsolve
from scipy.stats import iqr

pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader.wb as wb

# %%
def P(*args, **kwargs):
    P = np.linspace(-10, 10, 100).reshape(-1, 1)
    P = P[P != 0]
    return P


def AS(P=P(), W=0, P_e=1, Z_2=0):
    return P - Z_2


def AD(P=P(), M=0, G=0, T=0, Z_1=0):
    return -P + Z_1


# %%


def findIntersection(fun1, fun2, x0):
    return fsolve(lambda x: fun1(x) - fun2(x), x0)


indicators = wb.get_indicators()
indicators
# %%

# ### Get GDP Growth

indicators.loc[indicators.name == "GDP growth (annual %)", :]

countries = wb.get_countries()
countries.head()

# %%
# %%opts Curve [width=800, height=450]
gdp = wb.download(
    indicator="NY.GDP.MKTP.KD.ZG",
    country=["USA"],
    start=pd.to_datetime("1970", yearfirst=True),
    end=pd.to_datetime("2017", yearfirst=True),
)
gdp = gdp.reset_index().dropna()

gdp_unscaled = gdp

gdp.loc[gdp.country == "United States", "NY.GDP.MKTP.KD.ZG"] = (
    gdp.loc[gdp.country == "United States", "NY.GDP.MKTP.KD.ZG"]
    - gdp.loc[gdp.country == "United States", "NY.GDP.MKTP.KD.ZG"].mean()
) / iqr(gdp.loc[gdp.country == "United States", "NY.GDP.MKTP.KD.ZG"])

gdp_plot = gdp.iloc[::-1, :].hvplot.line(
    x="year", y="NY.GDP.MKTP.KD.ZG", by="country", title="GDP growth (annual %)"
)

gdp_plot
# %%

# ## OLS Model

from statsmodels.regression.linear_model import OLS
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn import linear_model
from sklearn.decomposition import PCA

import hvplot.pandas

dw = durbin_watson(pd.to_numeric(nyse_hist_10_years.Close).pct_change().dropna().values)
print(f"DW-statistic of {dw}")

# This strongly exceeds the upper-bound of the DW-statistic at the 5% level, indicating the presence of first order correlation.
# => Market is inefficient

# %%
# Public and monetary policies
# Students are required to investigate monetary, fiscal, and regulatory
# policies in the appropriate country and the institutional responses to
# changes brought about by the crisis. Students are
# required to discuss the challenges to traditional risk management
# strategies and financial models, given the concentration, liquidity and
# systematic risk factors and their impact on investor decisions.

# %%
# First, let's explore Liquidity Preference Curve before the COVID-19
# crisis from the following countries :
# - United States
# - France
# - Spain
# - Brazil
# - Russia

# %%
# c = countries.iso3c.sample(frac=0.1).tolist()
c = ["USA", "FRA", "RUS"]
# "ESP", "BRA", "RUS"]

money_lt = wb.download(
    indicator="FM.LBL.BMNY.GD.ZS",
    country=c,
    start=pd.to_datetime("1950", yearfirst=True),
    end=pd.to_datetime("2019", yearfirst=True),
).reset_index()

lending_rate_lt = wb.download(
    indicator="FR.INR.LEND",
    country=c,
    start=pd.to_datetime("1950", yearfirst=True),
    end=pd.to_datetime("2019", yearfirst=True),
).reset_index()

liquidity_trap_lt = money_lt.merge(lending_rate_lt, on=["year", "country"])
liquidity_trap_plot = liquidity_trap_lt.hvplot.scatter(
    x="FM.LBL.BMNY.GD.ZS", y="FR.INR.LEND", title="Liquidity Preference Data"
)
liquidity_trap_plot

# %%
# As we see from the graph above of United States, France, and Russia
# it's appear that a downward-sloping
# relationship
# do exists. We learnt from Module 5, that "a flattening of the Liquidity Preference Curve
# which renders Monetary Policy impotent, as change in Money Supply has limited effect
# on interest rates"

# In order to be certain, we will explore by modeling the data

liquidity_trap_lt.loc[:, "FM.LBL.BMNY.GD.ZS"] = np.log(
    liquidity_trap_lt.loc[:, "FM.LBL.BMNY.GD.ZS"]
)
liquidity_trap_lt.loc[:, "FR.INR.LEND"] = np.log(
    liquidity_trap_lt.loc[:, "FR.INR.LEND"]
)

import statsmodels.api as sm

liquidity_trap_lt.loc[:, "FM.LBL.BMNY.GD.ZS**0.5"] = liquidity_trap_lt.loc[
    :, "FM.LBL.BMNY.GD.ZS"
] ** (0.5)
liquidity_trap_lt.loc[:, "FM.LBL.BMNY.GD.ZS**2"] = liquidity_trap_lt.loc[
    :, "FM.LBL.BMNY.GD.ZS"
] ** (2)


# Compare the models with the inclusion of polynomial terms
#  and control for year
lt = pd.concat(
    [
        pd.get_dummies(liquidity_trap_lt.country),
        # pd.get_dummies(liquidity_trap_lt.year),
        liquidity_trap_lt.loc[:, "FM.LBL.BMNY.GD.ZS"],
        # liquidity_trap_lt.loc[:,'FM.LBL.BMNY.GD.ZS**0.5'],
        # liquidity_trap_lt.loc[:,'FM.LBL.BMNY.GD.ZS**2'],
        liquidity_trap_lt.loc[:, "FR.INR.LEND"],
    ],
    axis=1,
).dropna(0)

exo = sm.add_constant(lt.drop(columns=["FR.INR.LEND"]), prepend=False)

# # Fit and summarize OLS model
mod = sm.OLS(lt.loc[:, "FR.INR.LEND"], exo)
res = mod.fit()

res.summary()

# %%
# We can easily see that, the P value of the T test for Liquidity Trap is smaller
# than 0.005, which indicate a Liquidity Trap EXIST!
# Also, the Interest Rate coef is -1.5681, thus it support even more that
# before the crisis begin, Russia, United States, and France suffered from
# Liquidity Trap

# %% GDP
# GDP Plot
gdp = wb.download(
    indicator="NY.GDP.PCAP.KD",
    country=c,
    start=pd.to_datetime("2010", yearfirst=True),
    end=pd.to_datetime("2020", yearfirst=True),
)
gdp = gdp.reset_index()

gdp_plot = gdp.iloc[::-1, :].hvplot.line(
    x="year",
    y="NY.GDP.PCAP.KD",
    by="country",
    title="GDP per capita (constant 2010 US$)",
)
gdp_plot

# %%
