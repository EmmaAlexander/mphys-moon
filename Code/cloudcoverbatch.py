import pandas as pd
import reverse_geocode as rg
import numpy as np
from datetime import datetime,timedelta
from astropy.time import Time
from meteostat import Point, Hourly, Stations
import requests
from bs4 import BeautifulSoup
import re
import time
import pycountry_convert as pc
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="mphys-moon")
from geopy.extra.rate_limiter import RateLimiter
geocode = RateLimiter(geolocator.reverse, min_delay_seconds=1)


def switch(cont_code):
    #if (cont_code == "UK"): # UK is not a continent so needs adjusting
    #    return "ukuk&LAND=UK&KEY=UK"
    if (cont_code == "EU"):
        return "euro&LAND=__&KEY=__"
    elif (cont_code == "AF"):
        return "afri&LAND=__&KEY=__"
    elif (cont_code == "NA"):
        return "namk&LAND=__&KEY=__"
    elif (cont_code == "SA"):
        return "samk&LAND=__&KEY=__"
    elif (cont_code == "OC"):
        return "aupa&LAND=__&KEY=__"
    elif (cont_code == "AS"):
        return "asie&LAND=__&KEY=__"
    elif (cont_code == "AQ"):
        return "aris&LAND=__&KEY=__"

    
def continent_get(lon, lat):
    search_result = geocode((lat,lon))
    country_code = search_result.raw['address']['country_code']
    country_code = country_code.upper()

    if country_code=="UK" or country_code=="GB":
        return "ukuk&LAND=UK&KEY=UK"
    ca_countries=['BS','BZ','BM','KY','CR','CU','DO','SV','GT',
                  'HT','HN','JM','MX','NI','PA','PR','TC'] #list of central american countries counted by open weather
                    #does not include the lesser antilles islands
    if np.in1d(country_code, ca_countries)[0]:#add several centeral american countries
        return "mamk&LAND=__&KEY=__"
    
    continent_code = pc.country_alpha2_to_continent_code(country_code)
    return switch(continent_code)

def cloud_extract(date, lon, lat):
    # get UNIX date
    best_time = Time(date, format='jd') #best time in julian
    DATE = str(int(np.round(best_time.unix/60/30)*60*30)) #convert julian date to UNIX timestamp rounded to half hour
    if (int(DATE)<1612137600): # before accurate data
        return -1, -1

    #get url and download
    time.sleep(np.random.rand()/100) # sleep so as to not overwhelm the site
    CONT = continent_get(lon, lat)

    url = 'https://www.weatheronline.co.uk/weather/maps/current?LANG=en&DATE='+ DATE +'&CONT='+CONT+'&SORT=4&UD=0&INT=06&TYP=bedeckung&ART=tabelle&RUBRIK=akt&R=310&CEL=C&SI=mph'
    response = requests.get(url)

    if (response.status_code!=200): # website not accessed
        return -2, -2

    soup = BeautifulSoup(response.text, 'html.parser')

    #get table from website
    table = soup.find('table', class_ = 'gr3')
    table = str(table)

    #get names, codes and cloud level from table
    names = np.array(re.findall("(?<=CEL=C\">)(.*?)(?= \()", table))
    code = np.array(re.findall("(?<=WMO=)(.*?)(?=\&amp)", table))
    cloud_level = np.array(re.findall("(?<=<\/a><\/td>\n<td>)(.*)(?=<\/td>\n<t)", table))
    
    if (len(names)<7): #not enough datapoints
        return -3, -3
    
    cloud_valid = cloud_level!='not detectable'
    cloud_level = cloud_level[cloud_valid]
    code = code[cloud_valid]
    names = names[cloud_valid]
 
    #cloud_level = re.findall("(?<=<td>)(\d)(?=<\/td>)", table)

    # correct codes to agree with meteostat
    if "03931" in code: #tibenham
        code[np.where((code=="03931"))[0]] = "03546"
    if "03684" in code: #stansted
        code[np.where((code=="03684"))[0]] = "03683"
    if "03844" in code: #exeter
        code[np.where((code=="03844"))[0]] = "03839"

    day = best_time.to_datetime()
    #get 50 nearby stations
    stations = Stations().nearby(lat,lon)
    station = stations.fetch(50)
    #print(names)
    #print(station)

    #get the closest thats also online
    first_agree_index = np.where(np.in1d(station.index, code))[0]
    if len(first_agree_index) == 0:
        return -4, -4
    closest_code = station.iloc[[first_agree_index[0]]].index
    distance = station['distance'][first_agree_index[0]]

    cloud_for_point = cloud_level[np.where(np.in1d(code, closest_code))[0][0]]
    return int(cloud_for_point)/8, float(distance)

def pandas_cloud(df):
    print(df['Index'])
    return cloud_extract(df['Date'],df['Longitude'],df['Latitude'])

def main():
    LINUX = False
    data_file = 'Data\\moon_sighting_data.csv'
    if LINUX:
        data_file = '../Data/moon_sighting_data.csv'

    df = pd.read_csv(data_file, encoding="utf-8")
    #df = df[df["Source"]=="YALLOP"]
    #df = df.head(530)
    #df = df.tail(1)

    df['Cloud cover'], df['Distance'] = zip(*df.apply(pandas_cloud, axis=1))
    df = df[df['Cloud cover'] >= 0]
    df.to_csv('cloud_data2.csv', index=False)
    return 0

main()