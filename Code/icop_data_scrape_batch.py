import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import time
import random
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="mphys-moon")
from geopy.extra.rate_limiter import RateLimiter
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

years_to_search = [41,42,43,44,45]
islamic_months = ["muh","saf","raa","rat","jua","jut","raj","sha","ram","shw","kea","hej"]

def scrape_data(month_to_search):

    url = "https://www.astronomycenter.net/icop/"+month_to_search+".html?l=en"
    print(month_to_search)

    headers = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Encoding" : "gzip, deflate, br",
    "Accept-Language" : "en-GB,en;q=0.9",
    "Cache-Control" : "max-age=0",
    "Cookie" : "lang=en;",
    "Dnt" : "1",
    "Referer" : "https://www.astronomycenter.net/res.html",
    "User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
    }
    cookies = {"lang": "en"}
    response = requests.get(url, headers =headers, cookies=cookies)
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
    cloud_levels = re.findall("(?<=the sky was )(.*?)(?=, the atmospheric condition)",table)
    atmospheres = re.findall("(?<=atmospheric condition was )(.*?)(?=, the crescent)",table)
    v_eye = re.findall("(?<=the crescent was )(.*?)(?= by naked eye,)",table)
    v_bino = re.findall("(?<=naked eye, the crescent was )(.*?)(?= by binocular,)",table)
    v_telescope = re.findall("(?<=binocular, the crescent was )(.*?)(?= by telescope,)",table)
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
    data.to_csv("Data\\ICOP Updated\\"+month_to_search+".csv")

def run_batch():
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

def combine_batch():
    cols = ["Islamic Month","Date","City","State","Country","Obs Time","Cloud Level","Atmosphere" ,"Visibility","V Eye","V Binocular" ,"V Telescope" ,"V CCD"]
    data = pd.DataFrame(columns=cols)
    total = 0
    for year in years_to_search:
        if year == 41:
            search_months = ["kea","hej"]
        elif year == 45:
            search_months = ["muh","saf","raa","rat","jua"]
        else:
            search_months = islamic_months
        for month in search_months:
            month_to_search = month+str(year)
            print(month_to_search)
            
            d = pd.read_csv("Data\\ICOP Updated\\"+month_to_search+".csv")
            total+=d.shape[0]
            data = pd.concat([data,d])
    data.index.name = "Index"
    data = data[cols]
    data.to_csv("Data\\ICOP Updated\\icop2023_sighting_data_combined.csv")
    print(total)

def get_location(locname):
    loc = geocode(locname,language="en")
    if loc is None:
        adr = "Not found"
        lat = 0
        lon = 0
    else:
        adr = loc.address
        lat = loc.latitude
        lon = loc.longitude
    print(adr,lat,lon)
    return pd.Series([adr,lat,lon],index = ["Location", "Latitude", "Longitude"])

def get_locations():
    data = pd.read_csv("Data\\ICOP Updated\\icop2023_sighting_data_combined.csv",index_col=0)
    #data = data.head(5)

    data["Orig Location"] = data["City"] +"," +data["State"] +"," +data["Country"].str.strip()
    data["Orig Location"] = data["Orig Location"].str.strip("\"")
    location_data = data["Orig Location"].apply(get_location)
    data['Location'] = location_data["Location"]
    data['Latitude'] = location_data["Latitude"]
    data['Longitude'] = location_data["Longitude"]

    #df['point'] = df['location'].apply(lambda loc: tuple(loc.point) if loc else None)
    cols = ["Islamic Month","Date","Orig Location","Location","Latitude","Longitude","Obs Time","Cloud Level","Atmosphere" ,"Visibility","V Eye","V Binocular" ,"V Telescope" ,"V CCD"]
    data = data[cols]
    data.to_csv("Data\\icop2023_sighting_data_original.csv",index=False)
#combine_batch()
#get_locations()

def find_not_found():
        data = pd.read_csv("Data\\icop2023_sighting_data_original2.csv")
        missing_data = data[data['Location']=='Not found']
        print(f"{missing_data.shape[0]} out of {data.shape[0]}")
        print(f"{missing_data['Orig Location'].unique().shape[0]} unique out of {data.shape[0]}")
        #Code to generate unique missing locations - do not use as will replace missing file
        #data.to_csv("Data\\icop2023_sighting_data_original.csv")
        #missing_data_reduced = missing_data['Orig Location']
        #missing_data_reduced = pd.DataFrame(missing_data_reduced.unique())
        #missing_data_reduced.to_csv("Data\\icop2023_sighting_data_missing.csv")
find_not_found()


def replace_not_found():
        data = pd.read_csv("Data\\icop2023_sighting_data_original.csv")
        replacement =  pd.read_csv("Data\\icop2023_sighting_data_replace2.csv",index_col=0,sep=',',encoding = "utf-8")
        data.loc[data['Location']=='Not found','Longitude'] = replacement["Longitude"]
        data.loc[data['Location']=='Not found','Latitude'] = replacement["Latitude"]
        data.loc[data['Location']=='Not found','Location'] = replacement["Location"]

        #Code to generate unique missing locations - do not use as will replace missing file
        #data.to_csv("Data\\icop2023_sighting_data_original.csv")
        #missing_data_reduced = missing_data['Orig Location']
        #missing_data_reduced = pd.DataFrame(missing_data_reduced.unique())
        #missing_data_reduced.to_csv("Data\\icop2023_sighting_data_missing.csv")

        #replacement =  pd.read_csv("Data\\icop2023_sighting_data_replace.csv",index_col=0,sep=',',encoding = "utf-8")
        #replacement_locs = replacement["New"].apply(get_location)
        #replacement_locs.to_csv("Data\\icop2023_sighting_data_replace2.csv")
        data.to_csv("Data\\icop2023_sighting_data_original2.csv",index=False)
#replace_not_found()