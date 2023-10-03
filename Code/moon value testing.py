# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 11:52:00 2023

@author: Neil Power & Ezzy Cross
"""

#Imports
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import time
import random
from suncalc import get_times

from astropy.time import Time
from astropy.coordinates import Angle, EarthLocation, Longitude, Latitude
from astropy.coordinates import get_body, AltAz, solar_system_ephemeris
from astroplan import Observer
from astropy.constants import R_earth

#CALCULATING MOON Q-TEST VALUES -----------------------------------------------

def get_geocentric_parallax(object_position,distance):
    #Gets geocentric parallax of object
    #sin(p) = (a / r) sin(z')
    r = distance
    a = R_earth
    z = object_position.zen
    p = np.arcsin((a/r)*np.sin(z.radian))
    return Angle(p)

def get_q_value(alt_diff, width):
    #Calculate q-test value
    #q = (ARCV − (11·8371 − 6·3226 W' + 0·7319 W' 2 − 0·1018 W' 3 )) / 10

    ARCV = alt_diff
    W = width.arcmin
    q = (ARCV.deg - (11.8371 - 6.3226*W + 0.7319*W**2 - 0.1018*W**3 )) / 10

    return q


def get_best_obs_time(d,coords,display=False):
    #Gets best time using Bruin's method

    obs = Observer(location=coords, timezone="UTC")

    moonset=obs.moon_set_time(time=d,which='next',n_grid_points=150)
    sunset=obs.sun_set_time(time=d,which='next',n_grid_points=150)

    LAG = (moonset.to_value("jd")-sunset.to_value("jd"))*24*60

    if display:
        print(f"LAG: {LAG:.5} mins") #Lag time

    #Bruin best time Tb = (5 Ts +4 Tm)/ 9
    best_time = (1/9)*(5*sunset.to_value("jd")+4*moonset.to_value("jd"))

    return Time(best_time, format="jd")


def get_moon_params(d,lat,lon,time_given=False,display=False):
    #This calculates the q-test value and other moon params

    #Create coordinates object
    #Latitude is -90 to 90
    latitude = Latitude(lat*u.deg)
    #Longitude is -180 to +180
    longitude = Longitude(lon*u.deg,wrap_angle=180*u.deg)
    coords=EarthLocation.from_geodetic(lon=longitude,lat=latitude)

    #Calculate best observation time if no time given
    if not time_given:
        best_obs_time = get_best_obs_time(d, coords, display)
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
    ARCV = sun_altaz.alt - moon_altaz.alt

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

    if display:
        #Extra stuff

        #Calculate DAZ
        DAZ = sun_altaz.az - moon_altaz.az

        #Calculate moon semi-diameter and width
        W = SD*(1 - np.cos(ARCL.radian))

        #Calculate q-test value
        q = get_q_value(ARCV, W)

        #Cosine test: cos ARCL = cos ARCV cos DAZ
        cos_test = np.abs(np.cos(ARCL.radian)-np.cos(ARCV.radian)*np.cos(DAZ.radian))

        print(f"OBS LAT: {lat}. LON: {lon}")
        print(f"BEST OBS TIME: {best_obs_time.to_datetime().hour}:{best_obs_time.to_datetime().minute}")
        print(f"DATE: {d.to_value('datetime')}")
        print(f"JULIAN DATE: {d.to_value('jd')}")

        print(f"MOON ALT: {moon_altaz.alt:.4}. AZ: {moon_altaz.az:.4}")
        print(f"SUN ALT: {sun_altaz.alt:.4}. AZ: {sun_altaz.az:.4}")
        print(f"EARTH-MOON DIST: {MOON_EARTH_DIST:.2}")
        print(f"SUN-MOON DIST: {DIST:.2}")

        print(f"ARCL: {ARCL:.3}")
        print(f"ARCV: {ARCV:.3}")
        print(f"DAZ: {DAZ:.3}")
        print(f"PARALLAX: {parallax.arcmin:.3} arcmin")

        print(f"h: {h:.3}")
        print(f"W: {W.arcmin:.4} arcmin")
        print(f"W': {W_dash.arcmin:.4} arcmin")
        print(f"q(W): {q:.6}")
        print(f"q(W'): {q_dash:.6}")

        print(f"COS TEST: {cos_test:.4}")

        print()

    return q_dash


#Moon q-test value testing ---------------------------------------------------


#Example - first value of Yollop data (no 37), should produce ARCL=40.3, ARCV=31.1, DAZ=-26.1
#No time provided, usig best time which gives ARCL: 34.0, ARCV: 24.4, DAZ: 23.9
obs_date=Time("1870-7-25")
latitude = 38 #latitude in degrees
longitude = 23.7 #longitude in degrees

#get_moon_params(obs_date, latitude, longitude, display=True)


#Example - first value of Odeh data (no 514), produces ARCV=0.7, ARCL=5.8, DAZ=5.7 as expected
obs_date=Time("2452318.180",format='jd')
latitude = 43.9 #latitude in degrees
longitude = 18.4 #longitude in degrees

#get_moon_params(obs_date, latitude, longitude, time_given=True, display=True)

#Example - second value of Odeh data (no 565), also works
obs_date=Time("2452613.118",format='jd')
latitude = 30.9 #latitude in degrees
longitude = 35.8 #longitude in degrees

#get_moon_params(obs_date, latitude, longitude, time_given=True, display=True)

#PLOTTING MAP ----------------------------------------------------------------

def plot_visibilty_at_date(obs_date):
    #Plots a visibility graph at a specified date

    #lat/long over the globe
    lat_arr = np.linspace(-50, 50, 10)
    long_arr = np.linspace(-180, 180, 10)
    q_vals = np.zeros((len(lat_arr),len(long_arr)))

    start = time.time()
    for i in range(len(lat_arr)):
        lap = time.time()
        print(f"Calculating latitude {lat_arr[i]} at time={lap-start}")
        for j in range(len(long_arr)):
            q_vals[i,j] = get_moon_params(obs_date, lat_arr[i], long_arr[j])


    print("Total time:", time.time()-start)
    cont_plot(lat_arr,long_arr, q_vals)
    print(f"Max q: {np.max(q_vals)}. Min q: {np.min(q_vals)}")


def cont_plot(lat_arr,long_array,q_val):
    #Plots moon visibility across a world map
    x2, y2 = np.meshgrid(long_array,lat_arr,indexing='ij')
    plt.figure(figsize=(9,5))

    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    cs = plt.contourf(x2, y2 , q_val, levels = [-0.293,-0.232, -0.160, -0.014, +0.216],
                      alpha=0.3, cmap='brg' ,extend='both')
    plt.colorbar(cs)
    nm, lbl = cs.legend_elements()
    lbl_ = ['I(I)', 'I(V)', 'V(F)', 'V(V)', 'V']
    plt.legend(nm, lbl_)
    plt.xlabel(r'Latitude', fontsize=16)
    plt.ylabel(r'Longitude', fontsize=16)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.ylim(-90,90)
    plt.xlim(-180,180)
    #plt.legend()
    plt.title(r'Global moon visibility')
    #plt.xscale('log')
    #plt.yscale('log')
    plt.show()



obs_date = Time("2023-09-15")
#plot_visibilty_at_date(obs_date)

obs_date = Time("2023-09-16")
#plot_visibilty_at_date(obs_date)

obs_date = Time("2023-09-17")
#plot_visibilty_at_date(obs_date)

#Timing tests -----------------------------------------------

start = time.time()
for i in range(10): #VERY SLOW - 2s/10
    coords=EarthLocation.from_geodetic(lat=random.randint(-50,50),lon=random.randint(-180,180))
    get_best_obs_time(obs_date,coords)
print(time.time()-start)


start = time.time() #SLOW - 0.3s/10
for i in range(10):
    d=obs_date
    coords=EarthLocation.from_geodetic(lat=random.randint(-50,50),lon=random.randint(-180,180))
    with solar_system_ephemeris.set('builtin'):
        moon = get_body('moon',d,coords)
        sun = get_body('sun',d,coords)
        earth = get_body('earth',d,coords)
print(time.time()-start)



from skyfield import api

ts = api.load.timescale()
eph = api.load('de421.bsp')
from skyfield import almanac

start = time.time() #SLOW - 0.5s/10
for i in range(10):
    bluffton = api.wgs84.latlon(random.randint(-50,50),random.randint(-180,180))
    t0 = ts.utc(2023,9,16)
    t1 = ts.utc(2023,9,17)
    t, y = almanac.find_discrete(t0, t1, almanac.sunrise_sunset(eph, bluffton))

print(time.time()-start)

start = time.time() #SLOW - 0.5s/10
for i in range(10):
    bluffton = api.wgs84.latlon(random.randint(-50,50),random.randint(-180,180))
    t0 = ts.utc(2023,9,16)
    t1 = ts.utc(2023,9,17)
    f = almanac.risings_and_settings(eph, eph['Moon'], bluffton)
    t, y = almanac.find_discrete(t0, t1, f)

print(time.time()-start)

#
# start = time.time()
# lat_arr = np.linspace(-50, 50, 10)
# long_arr = np.linspace(-180, 180, 10)
# bluffton = api.wgs84.latlon(lat_arr,long_arr)

# t0 = ts.utc(2023,9,16)
# t1 = ts.utc(2023,9,17)
# f = almanac.risings_and_settings(eph, eph['Moon'], bluffton)
# t, y = almanac.find_discrete(t0, t1, f)
# print(time.time()-start)

start = time.time() #FAST - 0.06/10 and array option
for i in range(10):
    d=obs_date.to_datetime()
    get_times(d,lng=random.randint(-180,180),lat=random.randint(-50,50))["sunset"]

print(time.time()-start)
