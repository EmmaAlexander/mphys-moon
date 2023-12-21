#file to pull cloud cover data and generate a csv file of sightings with cloud cover

#imports
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

#function to convert continent codes to the url equivalent
def url_adjust(cont_code):
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

#function to get the continent from coordinates
def continent_get(lon, lat):
    #get the country code from the location
    search_result = geocode((lat,lon))
    country_code = search_result.raw['address']['country_code']
    country_code = country_code.upper()

    #the UK has its own url
    if country_code=="UK" or country_code=="GB":
        return "ukuk&LAND=UK&KEY=UK"
    
    #list of central american countries counted by weather online
    ca_countries=['BS','BZ','BM','KY','CR','CU','DO','SV','GT',
                  'HT','HN','JM','MX','NI','PA','PR','TC']
                    #does not include the lesser antilles islands, some territorys give misleading codes
    if np.in1d(country_code, ca_countries)[0]:
        return "mamk&LAND=__&KEY=__"
    
    #get continent from country
    continent_code = pc.country_alpha2_to_continent_code(country_code)
    return url_adjust(continent_code)

#take the cloud data from weather online
def cloud_extract(date, lon, lat):
    #get UNIX date for the URL
    best_time = Time(date, format='jd')
    DATE = str(int(np.round(best_time.unix/60/30)*60*30)) #convert julian date to UNIX timestamp rounded to half hour

    #skips sightings before consistant weather data
    if (int(DATE)<1612137600):
        return -1, -1

    CONT = continent_get(lon, lat)
    time.sleep(np.random.rand()/100) # sleep so as to not overwhelm the site

    #get the url
    url = 'https://www.weatheronline.co.uk/weather/maps/current?LANG=en&DATE='+ DATE +'&CONT='+CONT+'&SORT=4&UD=0&INT=06&TYP=bedeckung&ART=tabelle&RUBRIK=akt&R=310&CEL=C&SI=mph'
    response = requests.get(url)

    #catch if the url cannot be accessed
    if (response.status_code!=200):
        return -2, -2

    #get the html code for the page
    soup = BeautifulSoup(response.text, 'html.parser')

    #get table from page code
    table = soup.find('table', class_ = 'gr3')
    table = str(table)

    #get names, codes and cloud level from table
    names = np.array(re.findall("(?<=CEL=C\">)(.*?)(?= \()", table))
    code = np.array(re.findall("(?<=WMO=)(.*?)(?=\&amp)", table))
    cloud_level = np.array(re.findall("(?<=<\/a><\/td>\n<td>)(.*)(?=<\/td>\n<t)", table))
    
    #remove NaN cloud values
    cloud_valid = cloud_level!='not detectable'
    cloud_level = cloud_level[cloud_valid]
    code = code[cloud_valid]
    names = names[cloud_valid]

    # correct codes to agree with meteostat
    if "03931" in code: #tibenham
        code[np.where((code=="03931"))[0]] = "03546"
    if "03684" in code: #stansted
        code[np.where((code=="03684"))[0]] = "03683"
    if "03844" in code: #exeter
        code[np.where((code=="03844"))[0]] = "03839"

    day = best_time.to_datetime()
    #get 50 nearest stations
    stations = Stations().nearby(lat,lon)
    station = stations.fetch(50)

    #get the idex of the closest thats also online
    first_agree_index = np.where(np.in1d(station.index, code))[0]

    #catch if there is no agreement
    if len(first_agree_index) == 0:
        return -4, -4
    
    #get the code of the closesest station and the distance from the sighting
    closest_code = station.iloc[[first_agree_index[0]]].index
    distance = station['distance'][first_agree_index[0]]

    #find corresponding cloud level
    cloud_for_point = cloud_level[np.where(np.in1d(code, closest_code))[0][0]]
    return int(cloud_for_point)/8, float(distance)

#function to allow the pandas apply feature
def pandas_cloud(df):
    print(df['Index'])
    return cloud_extract(df['Date'],df['Longitude'],df['Latitude'])

def main():
    #read in the file
    data_file = '../Data/moon_sighting_data.csv'
    df = pd.read_csv(data_file, encoding="utf-8")

    #apply the cloud data gathering function
    df['Cloud cover'], df['Distance'] = zip(*df.apply(pandas_cloud, axis=1))

    #cut errored points and readings too far away
    df = df[df['Cloud cover'] >= 0]
    df = df[df['Distance'] <= 100000]
    df.to_csv('cloud_data_gen.csv', index=False)
    return 0

main()