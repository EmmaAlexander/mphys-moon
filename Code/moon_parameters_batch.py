# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 11:52:00 2023

@author: Neil Power & Ezzy Cross
"""

#Standard imports
from datetime import datetime
from zoneinfo import ZoneInfo

#Utility imports
import numpy as np
import pandas as pd

#Astropy imports
import astropy.units as u
from astropy.time import Time, TimeDelta
from astropy.coordinates import Angle, EarthLocation, Longitude, Latitude
from astropy.coordinates import get_body, AltAz, solar_system_ephemeris
from astropy.constants import R_earth

#Skyfield imports
from skyfield import api
ts = api.load.timescale()
eph = api.load('de421.bsp')
from skyfield import almanac

#Other imports
from suncalc import get_times
from astroplan import Observer
from timezonefinder import TimezoneFinder
ZoneFinder = TimezoneFinder()

#CALCULATING MOON PARAMETERS -----------------------------------------------

def get_geocentric_parallax(object_position,distance):
    #Gets geocentric parallax of object
    #sin(p) = (a / r) sin(z')
    r = distance
    a = R_earth
    z = object_position.zen
    p = np.arcsin((a/r)*np.sin(z.radian))
    return Angle(p)


def get_q_value(alt_diff, width):
    #Calculate q-test value using Yallop formula

    ARCV = alt_diff
    W = width.arcmin
    q = (ARCV.deg - (11.8371 - 6.3226*W + 0.7319*W**2 - 0.1018*W**3 )) / 10

    return q


def get_time_zone(latitude,longitude):
    #Returns an astropy timedelta object with correct UTC offset
    zone_name = ZoneFinder.timezone_at(lat=latitude,lng=longitude)
    utc_offset = datetime.now(ZoneInfo(zone_name)).utcoffset().total_seconds()
    return TimeDelta(utc_offset*u.second)

def get_best_obs_time(sunset, moonset):
    #Gets best time using Bruin's method

    sunset = Time(sunset, scale='utc')
    moonset = Time(moonset, scale='utc')

    #Bruin best time Tb = (5 Ts +4 Tm)/ 9
    best_time = (1/9)*(5*sunset.to_value("jd")+4*moonset.to_value("jd"))
    return Time(best_time, format="jd", scale='utc')


#CALCULATING SUNRISE/SUNSET TIMES----------------------------------------------

def get_sunset_time(obs_date, lat_arr,long_arr):
    #Gets sunset using suncalc (FAST, SUPPORTS ARRAYS)
    #Gets array of sunset times
    #Date needs to be Time object
    date_arr = np.full(np.size(lat_arr),obs_date.to_datetime())
    sunsets = get_times(date_arr,lng=long_arr,lat=lat_arr)["sunset"]
    return sunsets

def get_sunset_time2(d,lat,lon):
    #Gets sunset and moonset using astroplan (VERY SLOW)
    coords=EarthLocation.from_geodetic(lon=lon,lat=lat)
    
    obs = Observer(location=coords, timezone="UTC")

    #moonset=obs.moon_set_time(time=d,which='next',n_grid_points=150)
    sunset=obs.sun_set_time(time=d,which='next',n_grid_points=150)
    return sunset


def get_moonset_time(obs_date,lat, lon):
    #Gets moonset time using skyfield (MEDIUM)
    location = api.wgs84.latlon(lat,lon)

    t0 = ts.from_astropy(obs_date)
    t1 = ts.from_astropy(obs_date+TimeDelta(1,format="jd"))

    f = almanac.risings_and_settings(eph, eph['Moon'], location)
    t, y = almanac.find_discrete(t0, t1, f)

    moonsets = t[y==0]

    #MAY NO LONGER BE NEEDED?
    if len(moonsets) == 0: #If no moonset found, add another day to search forward
        t1 = ts.from_astropy(obs_date+TimeDelta(2,format="jd"))
        f = almanac.risings_and_settings(eph, eph['Moon'], location)
        t, y = almanac.find_discrete(t0, t1, f)
        moonsets = t[y==0]

    try:
        moonset = moonsets.utc_iso()[0]

    except:
        print(f"Error calculating moonset for Longitude: {lon} Latitude: {lat}.")
        moonset = obs_date

    return moonset

def get_sun_moonset_date(d,lat,lon):
    #Gets UTC time to search for moon/sunset from location
    twelve_hours = TimeDelta(0.5,format="jd")

    #Get time difference between UTC and local
    local_time_diff = get_time_zone(latitude=lat,longitude=lon)
    twelve_hours = TimeDelta(0.5,format="jd")

    #If west of 0 deg longitude, search from DD-1/MM/YYYY 12:00 LOCAL
    if lon < 0:
        #Create object that is  DD-1/MM/YYYY 12:00 LOCAL
        local_midday_day_before = d - twelve_hours

        #SUBTRACT time diff to go from local to UTC
        utc_search_time = local_midday_day_before - local_time_diff + TimeDelta(1,format="jd")


    #If east of 0 deg longitude, search from DD/MM/YYYY 00:00 UTC
    elif lon >= 0:
        #Create object that is  DD/MM/YYYY 12:00 LOCAL
        local_midday_day_of = d + twelve_hours

        #SUBTRACT time diff to go from local to UTC
        utc_search_time = local_midday_day_of - local_time_diff

    return utc_search_time

def get_new_moon_date(obs_date):
    #Gets date of last new moon
    one_month_ago = obs_date - TimeDelta(30,format="jd")
    t0 = ts.from_astropy(one_month_ago)
    t1 = ts.from_astropy(obs_date)
    t, y = almanac.find_discrete(t0, t1, almanac.moon_phases(eph))

    new_moon = t[y==0][-1] #Get most recent new moon
    new_moon_date = new_moon.utc_datetime().replace(hour=0, minute=0,second=0)
    return Time(new_moon_date)

def get_moon_age(obs_date):
    #Gets age of moon as a number of days
    last_new_moon = get_new_moon_date(obs_date)
    moon_age = obs_date-last_new_moon
    return moon_age.jd

#CALCULATE Q-VALUE
def get_moon_params(d,lat,lon):
    #This calculates the q-test value and other moon params

    #Create coordinates object
    #Latitude is -90 to 90
    latitude = Latitude(lat*u.deg)
    #Longitude is -180 to +180
    longitude = Longitude(lon*u.deg,wrap_angle=180*u.deg)
    coords=EarthLocation.from_geodetic(lon=longitude,lat=latitude)


    #Calculate sunset and moonset time
    sun_moonset_date = get_sun_moonset_date(d,lat,lon)
    #sunset = get_sunset_time(sun_moonset_date,lat,lon) #Use suncalc
    sunset = get_sunset_time2(sun_moonset_date,lat,lon) #Use astroplan
    moonset = get_moonset_time(sun_moonset_date,lat,lon)

    best_obs_time = get_best_obs_time(sunset, moonset)
    #Calculate best observation time if no time given
    d = best_obs_time

    #Get positions of Moon and Sun
    with solar_system_ephemeris.set('builtin'):
        moon = get_body('moon',d,coords)
        sun = get_body('sun',d,coords)
        earth = get_body('earth',d,coords)

    #Get moon and sun positions in alt/az coords
    moon_altaz = moon.transform_to(AltAz(obstime=d,location=coords))
    sun_altaz = sun.transform_to(AltAz(obstime=d,location=coords))

    #Get distance to Moon from Sun and Earth
    MOON_EARTH_DIST = moon.separation_3d(earth)
    DIST = moon.separation_3d(sun)

    #Calculate angular separation of moon and sun
    ARCL = moon.separation(sun)

    #Find alt/az difference
    ARCV = np.abs(sun_altaz.alt - moon_altaz.alt)

    #Calculate geocentric parallax
    parallax = get_geocentric_parallax(moon_altaz, MOON_EARTH_DIST)

    #Calculate moon altitude
    h = moon_altaz.alt

    #Calculate moon semi-diameter
    SD = 0.27245*parallax

    #Calculate topocentric moon semi-diameter and topcentric width
    SD_dash = SD*(1 + np.sin(h.radian)*np.sin(parallax.radian))
    W_dash = SD_dash*(1 - np.cos(ARCL.radian))

    #Calculate topocentric q-test value
    q_dash = get_q_value(ARCV, W_dash)

    #Extra stuff

    #Calculate DAZ
    DAZ = sun_altaz.az - moon_altaz.az

    #Calculate moon semi-diameter and width
    W = SD*(1 - np.cos(ARCL.radian))

    #Calculate q-test value
    q = get_q_value(ARCV, W)

    #Get moon age
    MOON_AGE = get_moon_age(d)
    LAG = (Time(moonset).to_value("jd")-Time(sunset).to_value("jd"))*24*60


    return np.round([d.to_value('jd'),
            lat,
            lon,
            round(MOON_AGE,3),
            Time(sunset).to_value("jd"),
            Time(moonset).to_value("jd"),
            LAG,
            moon_altaz.alt.deg,
            moon_altaz.az.deg,
            sun_altaz.alt.deg,
            sun_altaz.az.deg,
            MOON_EARTH_DIST.au,
            DIST.au,
            ARCL.deg,
            ARCV.deg,
            DAZ.deg,
            parallax.arcmin,
            W.arcmin,
            W_dash.arcmin,
            q,
            q_dash],decimals=5)

def cloud_replace(cloud_text):
    cloud_text = cloud_text.lower()
    if cloud_text == "clear":
        return 0
    elif cloud_text == "rainy":
        return 0.5
    elif cloud_text == "partly_cloudy":
        return 0.5
    elif cloud_text == "totally_cloudy":
        return 1
    else:
        print(f"Error with {cloud_text}")
        return -1

def read_and_update_file(file_name):
    raw_data = pd.read_csv(file_name)
    cols = ["Date",
            "Latitude",
            "Longitude",
            "Moon Age",
            "Sunset",
            "Moonset",
            "Lag",
            "Moon Alt",
            "Moon Az",
            "Sun Alt",
            "Sun Az",
            "Moon-Earth Dist",
            "Sun-Moon Dist",
            "ARCL",
            "ARCV",
            "DAZ",
            "Parallax",
            "W",
            "W'",
            "q",
            "q'",
            "Cloud Level",
            "Seen"]


    num_of_rows = raw_data.shape[0]
 
    data = pd.DataFrame(index=np.arange(0, num_of_rows), columns=cols)
    data.index.name="Index"
    for i, row in raw_data.iterrows():
        row_date = Time(datetime.strptime(row["Date"], "%d-%b-%y"))
        row_lat = float(row["Lat"])
        row_lon = float(row["Lon"])
        row_seen = row["Seen?"]
        row_cloud = cloud_replace(row["Clouds"])

        existing_data = [row_cloud, row_seen]
        new_data = get_moon_params(row_date,row_lat,row_lon)

        row_to_add = np.hstack((new_data,existing_data))
        data.loc[i] = row_to_add

        if i % 100 == 0:
            print(f"Generating row {i}")

    data.to_csv('..\\Data\\icouk_sighting_data_with_params.csv')


data_file = '..\\Data\\icouk_sighting_data.csv'
read_and_update_file(data_file)



