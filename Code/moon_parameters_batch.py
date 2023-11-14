# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 11:52:00 2023

@author: Neil Power & Ezzy Cross
"""

#Standard imports
import time
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
eph = api.load('de430_1850-2150.bsp')
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

def get_sunset_time(obs_date,lat, lon):
    #Gets moonset time using skyfield (MEDIUM)
    location = api.wgs84.latlon(lat,lon)

    t0 = ts.from_astropy(obs_date)
    t1 = ts.from_astropy(obs_date+TimeDelta(1,format="jd"))

    f = almanac.sunrise_sunset(eph, location)
    t, y = almanac.find_discrete(t0, t1, f)

    sunsets = t[y==0]

    #MAY NO LONGER BE NEEDED?
    if len(sunsets) == 0: #If no moonset found, add another day to search forward
        t1 = ts.from_astropy(obs_date+TimeDelta(2,format="jd"))
        f = almanac.sunrise_sunset(eph, location)
        t, y = almanac.find_discrete(t0, t1, f)
        sunsets = t[y==0]

    try:
        sunset = sunsets.utc_iso()[0]

    except:
        print(f"Error calculating sunset for Longitude: {lon} Latitude: {lat}.")
        sunset = obs_date

    return sunset

def get_sunset_time2(obs_date, lat_arr,long_arr):
    #Gets sunset using suncalc (FAST, SUPPORTS ARRAYS)
    #Gets array of sunset times
    #Date needs to be Time object
    date_arr = np.full(np.size(lat_arr),obs_date.to_datetime())
    sunsets = get_times(date_arr,lng=long_arr,lat=lat_arr)["sunset"]
    return sunsets

def get_sunset_time3(d,lat,lon):
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
    #sun_moonset_date = get_sun_moonset_date(d,lat,lon)
    #sunset = get_sunset_time(sun_moonset_date,lat,lon) #Use suncalc
    sunset = get_sunset_time(d,lat,lon) #Use astroplan
    moonset = get_moonset_time(d,lat,lon)

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

    #Calculate illumination
    ILLUMINATION = 0.5*(1 - np.cos(ARCL.radian))

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
            ILLUMINATION,
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
    elif cloud_text == "partly cloudy":
        return 0.75
    elif cloud_text == "totally_cloudy":
        return 1
    else:
        print(f"Error with {cloud_text}")
        return -1

def select_method_array(method):
    methods = []
    if method == "Not_seen":
        methods = ["Not_seen"]
    elif method == "Seen_ccd":
        methods = ["Seen_ccd"]
    elif method == "Seen_telescope":
        methods = ["Seen_telescope", "Seen_ccd"]
    elif method == "Seen_binoculars":
        methods = ["Seen_binoculars", "Seen_telescope", "Seen_ccd"]
    elif method == "Seen_eye":
        methods = ["Seen_eye", "Seen_binoculars", "Seen_telescope", "Seen_ccd"]
    return ";".join(methods)

def select_visibility_number(method):
    if method == "Not_seen":
        vis = 0
    elif method == "Seen_ccd":
        vis = 0.25
    elif method == "Seen_telescope":
        vis = 0.5
    elif method == "Seen_binoculars":
        vis = 0.75
    elif method == "Seen_eye":
        vis = 1
    return vis


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
            "Illumination",
            "Parallax",
            "W",
            "W'",
            "q",
            "q'",
            "Cloud Level",
            "Seen",
            "Method",
            "Methods",
            "Visibility"]

def select_method_ICOUK(row_seen,raw_method):
    if row_seen == "Not_seen":
        return row_seen
    elif raw_method == "Naked-eye":
        return "Seen_eye"
    elif raw_method == "Binoculars":
        return "Seen_binoculars"
    elif raw_method == "Naked-eye_and/or_Binoculars":
        return "Seen_eye"
    elif raw_method == "Binoculars_and_Naked-eye":
        return "Seen_eye"
    elif raw_method == "Naked-eye_and_Binoculars":
        return "Seen_eye"
    else:
        print(f"Error with {raw_method}")
        return -1

def read_and_update_file_ICOUK():
    data_file = '..\\Data\\icouk_sighting_data.csv'
    raw_data = pd.read_csv(data_file)

    num_of_rows = raw_data.shape[0]

    data = pd.DataFrame(index=np.arange(0, num_of_rows), columns=cols)
    data.index.name="Index"
    for i, row in raw_data.iterrows():
        row_date = Time(datetime.strptime(row["Date"], "%d-%b-%y"))
        row_lat = float(row["Lat"])
        row_lon = float(row["Lon"])
        row_seen = row["Seen?"]
        raw_method = row["Method"]
        row_method = select_method_ICOUK(row_seen,raw_method)
        row_methods = select_method_array(row_method)
        row_vis = select_visibility_number(row_method)
        row_cloud = cloud_replace(row["Clouds"])

        existing_data = [row_cloud, row_seen, row_method, row_methods,row_vis]
        new_data = get_moon_params(row_date,row_lat,row_lon)

        row_to_add = np.hstack((new_data,existing_data))
        data.loc[i] = row_to_add

        if i % 100 == 0:
            print(f"Generating row {i}")

    data["Source"] = np.full(data.shape[0],"ICOUK")
    data.to_csv('..\\Data\\icouk_sighting_data_with_params.csv')

def select_seen_ICOP(row_seene,row_seenb,row_seent):
    if row_seene:
        return "Seen"
    elif row_seenb:
        return "Seen"
    elif row_seent:
        return "Seen"
    else:
        return "Not_seen"

def select_method_ICOP(row_seene,row_seenb,row_seent,row_seenc):
    if row_seene:
        return "Seen_eye"
    elif row_seenb:
        return "Seen_binoculars"
    elif row_seent:
        return "Seen_telescope"
    elif row_seenc:
        return "Seen_ccd"
    else:
        return "Not_seen"


def read_and_update_file_ICOP():
    data_file = '..\\Data\\icop_ahmed_2020_sighting_data.csv'
    raw_data = pd.read_csv(data_file)

    num_of_rows = raw_data.shape[0]

    data = pd.DataFrame(index=np.arange(0, num_of_rows), columns=cols)
    data.index.name="Index"
    for i, row in raw_data.iterrows():
        row_date = Time(datetime.strptime(row["Date"], "%d/%m/%Y"))
        row_lat = float(row["lat"])
        row_lon = float(row["long"])
        row_seene = row["y_eye"]
        row_seenb = row["y_bino"]
        row_seent = row["y_tele"]
        row_seenc = row["y_ccd"]
        row_seen = select_seen_ICOP(row_seene,row_seenb,row_seent)
        row_method = select_method_ICOP(row_seene,row_seenb,row_seent,row_seenc)
        row_methods = select_method_array(row_method)
        row_vis = select_visibility_number(row_method)
        row_cloud = cloud_replace(row["x_sky"])

        existing_data = [row_cloud, row_seen, row_method, row_methods,row_vis]
        new_data = get_moon_params(row_date,row_lat,row_lon)

        row_to_add = np.hstack((new_data,existing_data))
        data.loc[i] = row_to_add

        #if i ==100:
        #    break

        if i % 100 == 0:
            print(f"Generating row {i}")
    data["Source"] = np.full(data.shape[0],"ICOP")
    data.to_csv('..\\Data\\icop_ahmed_2020_sighting_data_with_params.csv')

def select_means_alrefay(means):
    means = means.strip()
    if means == "N":
        return "Seen"
    elif means == "T":
        return "Seen"
    else:
        return "Not_seen"


def select_method_alrefay(means):
    means = means.strip()
    if means == "N":
        return "Seen_eye"
    elif means == "T":
        return "Seen_telescope"
    elif means == "Not_seen":
        return "Not_seen"
    else:
        print(f"Error with {means}")
        return -1

def read_and_update_file_alrefay():
    data_file = '..\\Data\\alrefay_2018_sighting_data.csv'
    raw_data = pd.read_csv(data_file)

    num_of_rows = raw_data.shape[0]

    data = pd.DataFrame(index=np.arange(0, num_of_rows), columns=cols)
    data.index.name="Index"
    for i, row in raw_data.iterrows():
        row_date = Time(datetime.strptime(row["Date"], " %Y/%m/%d"))
        row_lat = float(row["Lat."])
        row_lon = float(row["Long."])
        means = row["Means"]
        row_seen = select_means_alrefay(means)
        row_method = select_method_alrefay(means)
        row_methods = select_method_array(row_method)
        row_vis = select_visibility_number(row_method)
        row_cloud = 0

        existing_data = [row_cloud, row_seen, row_method, row_methods,row_vis]
        new_data = get_moon_params(row_date,row_lat,row_lon)

        row_to_add = np.hstack((new_data,existing_data))
        data.loc[i] = row_to_add

        #if i ==100:
        #    break

        if i % 100 == 0:
            print(f"Generating row {i}")

    data["Source"] = np.full(data.shape[0],"ALREFAY")
    data.to_csv('..\\Data\\alrefay_2018_sighting_data_with_params.csv')

def select_vis_allawi(vis):
    vis = vis.strip()
    if vis == "I":
        return "Not_seen"
    elif vis == "I(B)":
        return "Seen"
    elif vis == "I(T)":
        return "Seen"
    elif vis == "I(V)":
        return "Seen"
    elif vis == "V(F)":
        return "Seen"
    elif vis == "V":
        return "Seen"
    else:
        print(f"Error with {vis}")
        return -1

def select_method_allawi(vis):
    vis = vis.strip()
    if vis == "I":
        return "Not_seen"
    elif vis == "I(B)":
        return "Seen_binoculars"
    elif vis == "I(T)":
        return "Seen_telescope"
    elif vis == "I(V)":
        return "Seen_binoculars"
    elif vis == "V(F)":
        return "Seen_eye"
    elif vis == "V":
        return "Seen_eye"
    else:
        print(f"Error with {vis}")
        return -1


def read_and_update_file_allawi():
    data_file = 'mphys-moon/Data/schaefer_odeh_allawi_2022_sighting_data.csv'
    raw_data = pd.read_csv(data_file)

    num_of_rows = raw_data.shape[0]

    data = pd.DataFrame(index=np.arange(0, num_of_rows), columns=cols)
    data.index.name="Index"
    for i, row in raw_data.iterrows():
        date_text = row["Sight Date Best Time"].strip()[0:10]
        if date_text[2] == "-":
            row_date = Time(datetime.strptime(date_text, "%d-%m-%Y"))
        else:
            row_date = Time(datetime.strptime(date_text, "%Y-%m-%d"))
        row_lat = float(row["Lat"])
        row_lon = float(row["Lon"])
        visibility = row["SO"]
        row_seen = select_vis_allawi(visibility)
        row_method = select_method_allawi(visibility)
        row_methods = select_method_array(row_method)
        row_vis = select_visibility_number(row_method)
        row_cloud = 0

        existing_data = [row_cloud, row_seen, row_method, row_methods,row_vis]
        new_data = get_moon_params(row_date,row_lat,row_lon)

        row_to_add = np.hstack((new_data,existing_data))
        data.loc[i] = row_to_add

        if i % 100 == 0:
            print(f"Generating row {i}")

    data["Source"] = np.full(data.shape[0],"SCHAEFER/ODEH")
    data.to_csv('mphys-moon/Data/schaefer_odeh_allawi_2022_sighting_data_with_params.csv')


def select_vis_schaefer(vis):
    vis = vis.strip()
    if vis == "I": #Invisible with eye
        return "Not_seen"
    if vis == "I(I)": #Invisible with telescope and binoculars
        return "Not_seen"
    elif vis == "I(B)" : #Invisible with eye, and binoculars
        return "Not_seen"
    elif vis == "I(T)": #Invisible with eye, and with telescope
        return "Not_seen"
    elif vis == "I(V)": #Invisible with eye and with either binoculars or telescope
        return "Not_seen"
    elif vis == "V(T)": #Not visible with eye, visible with binoculars
        return "Seen"
    elif vis == "V(B)": #Not visible with eye visible with telescope
        return "Seen"
    elif vis == "V(V)": #Not visible with eye visible with either binoculars or telescope
        return "Seen"
    elif vis == "V(F)": #Visible with eye, after locating with visual aid
        return "Seen"
    elif vis == "V": #Visible with eye
        return "Seen"
    else:
        print(f"Error with {vis}")
        return -1
    
def select_method_schaefer(vis):
    vis = vis.strip()
    if vis == "I": #Invisible with eye
        return "Not_seen"
    if vis == "I(I)": #Invisible with telescope and binoculars
        return "Not_seen"
    elif vis == "I(B)" : #Invisible with eye, and binoculars
        return "Not_seen"
    elif vis == "I(T)": #Invisible with eye, and with telescope
        return "Not_seen"
    elif vis == "I(V)": #Invisible with eye and with either binoculars or telescope
        return "Not_seen"
    elif vis == "V(F)": #Visible with eye, after locating with visual aid
        return "Seen_eye" #Assuming seen
    elif vis == "V(B)": #Not visible with eye, visible with binoculars
        return "Seen_binoculars"
    elif vis == "V(T)": #Not visible with eye visible with telescope
        return "Seen_telescope"
    elif vis == "V(V)": #Not visible with eye visible with either binoculars or telescope
        return "Seen_binoculars" #Assuming binoculars
    elif vis == "V": #Visible
        return "Seen_eye"
    else:
        print(f"Error with {vis}")
        return -1

def read_and_update_file_yallop():
    data_file = 'Data\\yallop_sighting_data.csv'
    raw_data = pd.read_csv(data_file)

    num_of_rows = raw_data.shape[0]

    data = pd.DataFrame(index=np.arange(0, num_of_rows), columns=cols)
    data.index.name="Index"
    for i, row in raw_data.iterrows():
        date_text = f"{row['D']}-{row['M']}-{row['Y']}"
        row_date = Time(datetime.strptime(date_text, "%d-%m-%Y"))

        row_lat = float(row["Lat"])
        row_lon = float(row["Long"])
        visibility = row["Visibility"]
        row_seen = select_vis_schaefer(visibility)
        row_method = select_method_schaefer(visibility)
        row_methods = select_method_array(row_method)
        row_vis = select_visibility_number(row_method)
        row_cloud = 0

        existing_data = [row_cloud, row_seen, row_method, row_methods,row_vis]
        new_data = get_moon_params(row_date,row_lat,row_lon)

        row_to_add = np.hstack((new_data,existing_data))
        data.loc[i] = row_to_add

        if i % 100 == 0:
            print(f"Generating row {i}")

    data["Source"] = np.full(data.shape[0],"YALLOP")

    data.to_csv('Data\\yallop_sighting_data_with_params.csv')


def generate_parameters(date,min_lat, max_lat, min_lon, max_lon,no_of_points):

    num_of_rows = no_of_points*no_of_points

    data = pd.DataFrame(index=np.arange(0, num_of_rows), columns=cols[0:-5])
    data.index.name="Index"

    lat_arr = np.linspace(min_lat, max_lat, no_of_points)
    lon_arr = np.linspace(min_lon, max_lon, no_of_points)

    start = time.time()
    for i, latitude in enumerate(lat_arr):
        lap = time.time()
        print(f"Calculating latitude {round(lat_arr[i],2)} at time={round(lap-start,2)}s")
        for j, longitude in enumerate(lon_arr):
        
            data.loc[i] = get_moon_params(date,latitude,longitude)
        
    data.to_csv(f'mphys-moon/Data/Generated/{date.to_datetime().date()} LAT {min_lat} {max_lat} LON {min_lon} {max_lon} {no_of_points}x{no_of_points}.csv')
    print(f"Total time: {round(time.time()-start,2)}s")

#read_and_update_file_ICOUK()

#read_and_update_file_ICOP()

#read_and_update_file_alrefay()

read_and_update_file_allawi()

#date_to_use = Time("2023-03-22")
#generate_parameters(date_to_use,min_lat=-60, max_lat=60, min_lon=-180, max_lon=180, no_of_points=40)

#add_sources()

# raw_data = pd.read_csv('Data\\schaefer_odeh_allawi_2022_sighting_data_with_params.csv')
# raw_data["Date"] = Time(raw_data["Date"],format="jd").to_datetime()
# raw_data.to_csv('Data\\schaefer_odeh_allawi_2022_sighting_data_with_params2.csv')