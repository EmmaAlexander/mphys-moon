# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 11:52:00 2023

@author: Neil Power
"""

#Imports
import numpy as np
import astropy.units as u

from astropy.time import Time
from astropy.coordinates import  SkyCoord, EarthLocation, get_body, AltAz, solar_system_ephemeris
from astroplan import Observer



#Example - final value of Yollop data, should produce ARCL=5.5, ARCV=4.2, DAZ=3.6
d=Time("1984-01-03 05:15") #5:15
lat=15.6 #latitude in degrees
lon=35.6 #longitude in degrees
coords=EarthLocation.from_geodetic(lon*u.deg,lat*u.deg)
print("Date:", d)

#Testing - code correctly shows differnce in azimuth
# obs = Observer(location=coords, timezone="UTC")
# print("Getting sunset and moonset times for specified date:")
# moonset=obs.moon_set_time(time=d,which='next')
# sunset=obs.sun_set_time(time=d,which='next')
# print(moonset,sunset)


#Get positions of Moon and Sun
with solar_system_ephemeris.set('builtin'):    
    moon = get_body('moon',d,coords)
    sun = get_body('sun',d,coords)
    
    
#Get distance to Sun from moon
DIST =moon.separation_3d(sun)
print(f"DIST: {DIST:.2}")   
    
#Calclate angular separation of moon and sun
ARCL = moon.separation(sun)
print(f"ARCL: {ARCL.degree:.2}")


#Get moon and sun positions in alt/az coords
moon_altaz=moon.transform_to(AltAz(obstime=d,location=coords))
sun_altaz=sun.transform_to(AltAz(obstime=d,location=coords))

print(f"Moon altitude: {moon_altaz.alt.degree:.4}")
print(f"Moon azimuth: {moon_altaz.az.degree:.4}")

print(f"Sun altitude: {sun_altaz.alt.degree:.4}")
print(f"Sun azimuth: {sun_altaz.az.degree:.4}")

#Find difference
ARCV = np.abs( moon_altaz.alt - sun_altaz.alt)
DAZ = np.abs(moon_altaz.az - sun_altaz.az)

print(f"ARCV: {ARCV.degree:.2}")
print(f"DAZ: {DAZ.degree:.2}")


#Work in progress
parallax = 1
h = 1

print("Parallax: ",parallax )
print("h: ",h )

SD = 0.27245*parallax*(1+np.sin(h)*np.sin(np.deg2rad(parallax)))

W = SD*(1-np.cos(ARCL.radian))

print(f"W': {W:.4}")
q = (ARCV.radian - (11.8371 - 6.3226*W + 0.7319*W**2 - 0.1018*W**3 )) / 10
print(f"q: {q:.4}")

