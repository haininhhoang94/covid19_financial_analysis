# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

#%%
# import library
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

#%%
# get stock info by Yahoo!Finance
# data will get from 7/31/2019 to 7/31/2020

# path
p = "/home/haininhhoang94/WSL/Projects/covid19_financial_analysis"
while True:
    try:
        nyse_hist = pd.read_csv(p + "/Data/NYSE.csv")
        break
    except FileNotFoundError:
        nyse = yf.Ticker("^NYA")
        nyse.info
        nyse_hist = nyse.history(period="365d")
        nyse_hist.to_csv(p + "/Data/NYSE.csv")

#%%
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

        deaths = pd.DataFrame.from_dict(
            all_location_json["timelines"]["deaths"]["timeline"], orient="index",
        ).rename(columns={0: "{}_deaths".format(countries[i])})

        # recovered = pd.DataFrame.from_dict(
        #     all_location_json["timelines"]["recovered"]["timeline"], orient="index",
        # ).rename(columns={0: "{}_recovered".format(countries[i])})

        if i == 0:
            covid_df = confirmed
        else:
            covid_df = covid_df.join(confirmed)
        covid_df = covid_df.join(deaths)
        # covid_df = covid_df.join(recovered)
    return covid_df


countries = ["US", "FR"]
covid_df = covid_19_data_collection(countries)
#%%
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


unemployment_df = unemployment_rate(2000, 2020)
#%%
#%%
