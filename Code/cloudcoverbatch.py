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

icouk_data_file = 'Data\\icouk_sighting_data_with_params.csv'
df = pd.read_csv(icouk_data_file)

def cloud_extract(date, lon, lat):
    # get UNIX date
    best_time = Time(date, format='jd') #best time in julian
    DATE = str(int(np.round(best_time.unix/60/30)*60*30)) #convert julian date to UNIX timestamp rounded to half hour

    if (int(DATE)<1612137600): # before accurate data
        return -1

    #get url and download
    time.sleep(np.random.rand()/5) # sleep so as to not overwhelm the site
    url = 'https://www.weatheronline.co.uk/weather/maps/current?LANG=en&DATE='+ DATE +'&CONT=ukuk&LAND=UK&KEY=UK&SORT=4&UD=0&INT=06&TYP=bedeckung&ART=tabelle&RUBRIK=akt&R=310&CEL=C&SI=mph'
    response = requests.get(url)

    if (response.status_code!=200): # website not accessed
        return -2

    soup = BeautifulSoup(response.text, 'html.parser')

    #get table from website
    table = soup.find('table', class_ = 'gr3')
    table = str(table)

    #get names, codes and cloud level from table
    names = re.findall("(?<=CEL=C\">)(.*?)(?= \()", table)
    code = re.findall("(?<=WMO=)(.*?)(?=\&amp)", table)
    cloud_level = re.findall("(?<=<td>)(\d)(?=<\/td>)", table)
    
    if (len(names)<7): #not enough datapoints
        return -3

    day = best_time.to_datetime()
    #get 10 nearby stations that had a reading in the hourly interval
    stations = Stations().nearby(lat,lon)
    stations = stations.inventory('hourly', day)
    station = stations.fetch(50)

    #get the closest thats also online
    #closest_name = station['name'][np.where(np.in1d(station.index, code))[0][0]]
    closest_code = station.iloc[[np.where(np.in1d(station.index, code))[0][0]]].index
    #cloud_for_point = cloud_level[np.where(np.in1d(names, closest_name))[0][0]]
    cloud_for_point = cloud_level[np.where(np.in1d(code, closest_code))[0][0]]
    return int(cloud_for_point)

def pandas_cloud(df):
    print(df['Index'])
    return cloud_extract(df['Date'],df['Longitude'],df['Latitude'])

#x = cloud_extract(df['Date'][600],df['Longitude'][600],df['Latitude'][600])
x = df.apply(pandas_cloud, axis=1)
x.to_csv('cloudtest.csv')