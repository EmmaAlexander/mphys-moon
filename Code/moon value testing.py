# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 11:52:00 2023

@author: Neil Power & Ezzy Cross
"""

#Imports
import numpy as np
import astropy.units as u

from astropy.time import Time
from astropy.coordinates import EarthLocation, get_body, AltAz, solar_system_ephemeris
from astroplan import Observer

def get_best_obs_time(d,coords):
    #Gets best time using Bruin's method
    obs = Observer(location=coords, timezone="UTC")
    moonset=obs.moon_set_time(time=d,which='next')
    sunset=obs.sun_set_time(time=d,which='next')
    LAG = (moonset.to_value("jd")-sunset.to_value("jd"))*24*60
    print(f"LAG: {LAG:.5} mins") #Lag time

    #Bruin best time Tb = (5 Ts +4 Tm)/ 9
    best_time = (1/9)*(5*sunset.to_value("jd")+4*moonset.to_value("jd"))

    #Returns as Time object
    return Time(best_time,format="jd")


def get_moon_params(d,lat,lon,time_given=False):
    coords=EarthLocation.from_geodetic(lon=lon*u.deg,lat=lat*u.deg)
    print(f"OBS LAT: {lat}. LON: {lon}")

    best_obs_time = get_best_obs_time(d, coords).to_datetime()
    print(f"BEST OBS TIME: {best_obs_time.hour}:{best_obs_time.minute}")

    if not time_given:
        d = Time(best_obs_time)

    print(f"DATE: {d.to_value('datetime')}")
    print(f"JULIAN DATE: {d.to_value('jd')}")


    #Get positions of Moon and Sun
    with solar_system_ephemeris.set('builtin'):
        moon = get_body('moon',d,coords)
        sun = get_body('sun',d,coords)
        earth = get_body('earth',d,coords)

    #Get moon and sun positions in alt/az coords
    moon_altaz = moon.transform_to(AltAz(obstime=d,location=coords))
    sun_altaz = sun.transform_to(AltAz(obstime=d,location=coords))

    print(f"MOON ALT: {moon_altaz.alt:.4}. AZ: {moon_altaz.az:.4}")
    print(f"SUN ALT: {sun_altaz.alt:.4}. AZ: {sun_altaz.az:.4}")

    #Get distance to Sun from moon
    DIST = moon.separation_3d(sun)
    print(f"SUN-MOON DIST: {DIST:.2}")

    #Calclate angular separation of moon and sun
    ARCL = moon.separation(sun)
    print(f"ARCL: {ARCL:.3}")

    #Find alt/az difference
    ARCV = np.abs( moon_altaz.alt - sun_altaz.alt)
    DAZ = np.abs(moon_altaz.az - sun_altaz.az)

    print(f"ARCV: {ARCV:.3}")
    print(f"DAZ: {DAZ:.3}")


    #Parallax
    parallax = DIST.parallax
    print(f"PARALLAX: {parallax.degree:.3} deg")

    #Work in progress - should maybe be absolute value?
    h = moon_altaz.alt
    print(f"h: {h:.3}")

    SD = 0.27245*parallax.arcmin*(1+np.sin(h.radian)*np.sin(parallax.radian))

    W = SD*(1-np.cos(ARCL.radian))

    print(f"W': {W:.4}")
    q = (ARCV.radian - (11.8371 - 6.3226*W + 0.7319*W**2 - 0.1018*W**3 )) / 10
    print(f"q: {q:.6}")

    #cos ARCL = cos ARCV cos DAZ
    cos_test = np.abs(np.cos(ARCL.radian)-np.cos(ARCV.radian)*np.cos(DAZ.radian))
    print(f"COS TEST: {cos_test:.4}")

    print()


#Example - first value of Yollop data (no 37), should produce ARCL=40.3, ARCV=31.1, DAZ=-26.1
#No time provided, usig best time which gives ARCL: 34.0, ARCV: 24.4, DAZ: 23.9
obs_date=Time("1870-7-25")
latitude = 38 #latitude in degrees
longitude = 23.7 #longitude in degrees

get_moon_params(obs_date, latitude, longitude)


#Example - first value of Odeh data (no 514), produces ARCV=0.7, ARCL=5.8, DAZ=5.7 as expected
obs_date=Time("2452318.180",format='jd')
latitude = 43.9 #latitude in degrees
longitude = 18.4 #longitude in degrees

#get_moon_params(obs_date, latitude, longitude,time_given=True)

#Example - second value of Odeh data (no 565), also works
obs_date=Time("2452613.118",format='jd')
latitude = 30.9 #latitude in degrees
longitude = 35.8 #longitude in degrees

#get_moon_params(obs_date, latitude, latitude,time_given=True)

