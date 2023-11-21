import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import time
import random

years_to_search = [41,42,43,44,45]
islamic_months = ["muh","saf","raa","rat","jua","jut","raj","sha","ram","shw","kea","hej"]
last_recorded = "shw41"
current = "jua45"
month_to_search = "jua45"

def scrape_data(month_to_search):

    url = "https://www.astronomycenter.net/icop/"+month_to_search+".html"

    headers = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Encoding" : "gzip, deflate, br",
    "Accept-Language" : "en-GB,en;q=0.9",
    "Cache-Control" : "max-age=0",
    "Cookie" : "ASP.NET_SessionId=skottm30rlapyeln5m1mrgq2; lang=en; _gid=GA1.2.809394619.1700578346; _ga=GA1.2.421384393.1700578346; _ga_1LJHW8NDRE=GS1.1.1700578345.1.1.1700579677.47.0.0",
    "Dnt" : "1",
    "Referer" : "https://www.astronomycenter.net/res.html",
    "User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers =headers)
    print(response.status_code)
    soup = BeautifulSoup(response.text, 'html.parser')

    table = soup.find('div', class_ = 'container obs')
    table = str(table)
    countries_h2 = soup.find_all('h2',class_='')[2:-1] #Country elements
    country_lines = np.array([c.sourceline for c in countries_h2])

    dates_h2 = soup.find_all('h2',class_='required') #Date elements
    date_lines = np.array([d.sourceline for d in dates_h2])

    table = str(table)

    observation_times = []
    visibilities = []
    countries = []
    obs_dates = []
    vis_info = soup.find_all('div', class_ = 'observ') # More than 1 observ per country
    for observation in vis_info:

        #Need to get last country_lines number less than value
        observation_line = observation.sourceline
        corresponding_country_index = np.where(country_lines <  observation_line)[0][-1]
        country = countries_h2[corresponding_country_index].text

        #Need to get last country_lines number less than value
        corresponding_date_index = np.where(date_lines <  observation_line)[0][-1]
        date = dates_h2[corresponding_date_index].text

        spans = observation.find_all('span')
        observation_time = spans[0].text.strip()
        visibility = spans[1].text.strip()

        observation_times.append(observation_time)
        visibilities.append(visibility)
        countries.append(country)
        obs_dates.append(date)

    cities = re.findall("(?<=from )(.*?)(?= City)",table)
    states = re.findall("(?<=City in )(.*?)(?= State)",table)
    cloud_levels = re.findall("(?<=the sky was )(.*?)(?=,)",table)
    atmospheres = re.findall("(?<=atmospheric condition was )(.*?)(?=,)",table)
    v_eye = re.findall("(?<=the crescent was )(.*?)(?= by naked eye)",table)
    v_bino = re.findall("(?<=naked eye, the crescent was )(.*?)(?= by binocular)",table)
    v_telescope = re.findall("(?<=binocular, the crescent was )(.*?)(?= by telescope)",table)
    v_ccd = re.findall("(?<=telescope, the crescent was )(.*?)(?= by CCD Imaging)",table)
    islamic_dates = np.full(len(countries), month_to_search)

    initial_data = {"Islamic Month": islamic_dates,
                    "Date": obs_dates,
                    "City": cities,
                    "State": states,
                    "Country":countries,
                    "Obs Time": observation_times,
                    "Cloud Level": cloud_levels,
                    "Atmosphere" : atmospheres,
                        "Visibility":visibilities,
                        "V Eye": v_eye,
                        "V Binocular": v_bino ,
                            "V Telescope": v_telescope ,
                            "V CCD": v_ccd }
    data = pd.DataFrame(initial_data)
    data.to_csv("..\\Data\\ICOP Updated\\"+month_to_search+".csv")

for year in years_to_search:
    if year == 41:
        search_months = ["kea","hej"]
    elif year == 45:
        search_months = ["muh","saf","raa","rat","jua"]
    else:
        search_months = islamic_months
    for month in search_months:
        month_to_search = month+str(year)
        scrape_data(month_to_search)
        time.sleep(2+random.randint(1,3))