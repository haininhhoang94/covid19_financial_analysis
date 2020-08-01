# get data covid_19
# while True:
#     try:
#         with open(p + "/Data/all_location.txt") as outfile:
#             all_location_json = json.load(outfile)
#         break
#     except FileNotFoundError:
#         covid = cpy.COVID19(url="https://covid-tracker-us.herokuapp.com")
#         # US_covid = covid.getLocationByCountryCode("US", timelines=True)
#         # vir = dict(US_covid["latest"])
#         # all_location_json = covid.getAll(timelines=True)
#         all_location_json = covid.getLocationByCountryCode("US")(timelines=True)
#         with open(p + "/Data/all_location.txt", "w") as outfile:
#             json.dump(all_location_json, outfile)

