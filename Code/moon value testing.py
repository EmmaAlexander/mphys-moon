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

from astropy.time import Time
from astropy.coordinates import Angle, Distance, EarthLocation
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
    moonset=obs.moon_set_time(time=d,which='next')
    sunset=obs.sun_set_time(time=d,which='next')
    LAG = (moonset.to_value("jd")-sunset.to_value("jd"))*24*60
    if display:
        print(f"LAG: {LAG:.5} mins") #Lag time

    #Bruin best time Tb = (5 Ts +4 Tm)/ 9
    best_time = (1/9)*(5*sunset.to_value("jd")+4*moonset.to_value("jd"))

    #Returns as Time object
    return Time(best_time,format="jd")


def get_moon_params(d,lat,lon,time_given=False,display=False):
    #This calculates the q-test value and other moon params

    #Create coordinates object
    coords=EarthLocation.from_geodetic(lon=lon*u.deg,lat=lat*u.deg)

    #Calculate best observation time if no time given
    if not time_given:
        best_obs_time = get_best_obs_time(d, coords,display).to_datetime()
        d = Time(best_obs_time)

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
    DAZ = sun_altaz.az - moon_altaz.az

    #Calculate geocentric parallax
    parallax = get_geocentric_parallax(moon_altaz, MOON_EARTH_DIST)

    #Calculate moon altitude
    h = moon_altaz.alt

    #Calculate moon semi-diameter and topcentric width
    SD = 0.27245*parallax
    W = SD*(1 - np.cos(ARCL.radian))

    #Calculate topocentric moon semi-diameter and topcentric width
    SD_dash = SD*(1 + np.sin(h.radian)*np.sin(parallax.radian))
    W_dash = SD_dash*(1 - np.cos(ARCL.radian))

    #Calculate q-test value
    q = get_q_value(ARCV, W)

    #Calculate topocentric q-test value
    q_dash = get_q_value(ARCV, W_dash)

    #Cosine test: cos ARCL = cos ARCV cos DAZ
    cos_test = np.abs(np.cos(ARCL.radian)-np.cos(ARCV.radian)*np.cos(DAZ.radian))

    if display:
        print(f"OBS LAT: {lat}. LON: {lon}")
        print(f"BEST OBS TIME: {best_obs_time.hour}:{best_obs_time.minute}")
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
        print(f"W: {W:.4} arcmin")
        print(f"W': {W_dash:.4} arcmin")
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

get_moon_params(obs_date, latitude, longitude, display=True)


#Example - first value of Odeh data (no 514), produces ARCV=0.7, ARCL=5.8, DAZ=5.7 as expected
obs_date=Time("2452318.180",format='jd')
latitude = 43.9 #latitude in degrees
longitude = 18.4 #longitude in degrees

#get_moon_params(obs_date, latitude, longitude, time_given=True, display=True)

#Example - second value of Odeh data (no 565), also works
obs_date=Time("2452613.118",format='jd')
latitude = 30.9 #latitude in degrees
longitude = 35.8 #longitude in degrees

#get_moon_params(obs_date, latitude, latitude, time_given=True, display=True)

#PLOTTING MAP ----------------------------------------------------------------

#lat/long over the globe
lat_arr = np.linspace(-90, 90, 1000)
long_arr = np.linspace(-90, 90, 1000)

q_vals = np.zeros((len(lat_arr),len(long_arr)))
level_arr = np.linspace(0, 400, 400)

for i in range(len(long_arr)):
    for j in range(len(lat_arr)):
        temp_q = -0.16+(lat_arr[i]-long_arr[j])/100
        q_vals[j,i] = temp_q

#function to plot moon visibility across a world map
def cont_plot(lat_arr,long_array,q_val):

    x2, y2 = np.meshgrid(lat_arr, long_array, indexing='ij')
    plt.figure()

    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    cs = plt.contourf(x2, y2 , q_val, levels = [-0.293,-0.232, -0.160, -0.014, +0.216],
                      alpha=0.3, cmap='brg' ,extend='both')
    plt.colorbar(cs)
    nm, lbl = cs.legend_elements()
    lbl_ = ['I(I)', 'I(V)', 'V(F)', 'V(V)', 'V']
    plt.legend(nm, lbl_)
    plt.xlabel(r'Lattitude', fontsize=16)
    plt.ylabel(r'Longitude', fontsize=16)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    #plt.xlim(0.3,0.7)
    #plt.ylim(10,12)
    #plt.legend()
    plt.title(r'Moon visability across the globe', y=1.03)
    #plt.xscale('log')
    #plt.yscale('log')
    plt.show()

cont_plot(lat_arr,long_arr, q_vals)