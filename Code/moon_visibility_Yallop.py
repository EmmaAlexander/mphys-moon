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
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

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

#Cartopy imports
import cartopy.crs as ccrs
from cartopy import feature as cfeature

#Geovista imports
import geovista as gv

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

    #Note -  some slight discrepancy between T_S + 4/9 LAG and below method
    #lag = sunset.to_value("jd") - moonset.to_value("jd")
    #best_time = sunset + (4/9)*TimeDelta(lag, format="jd")
    #return best_time

    #Bruin best time Tb = (5 Ts +4 Tm)/ 9
    best_time = (1/9)*(5*sunset.to_value("jd")+4*moonset.to_value("jd"))
    return Time(best_time, format="jd", scale='utc')


#CALCULATING SUNRISE/SUNSET TIMES----------------------------------------------

def get_sunset_time_old(obs_date, lat_arr,long_arr):
    #Gets sunset using suncalc (FAST, SUPPORTS ARRAYS)
    #Gets array of sunset times
    #Date needs to be Time object
    date_arr = np.full(np.size(lat_arr),obs_date.to_datetime())
    sunsets = get_times(date_arr,lng=long_arr,lat=lat_arr)["sunset"]
    return sunsets

def get_moonset_time(obs_date,lat, lon,moonrise=False):
    #Gets moonset time using skyfield (MEDIUM)
    location = api.wgs84.latlon(lat,lon)

    t0 = ts.from_astropy(obs_date)
    t1 = ts.from_astropy(obs_date+TimeDelta(1,format="jd"))

    f = almanac.risings_and_settings(eph, eph['Moon'], location)
    t, y = almanac.find_discrete(t0, t1, f)

    moonsets = t[y==moonrise]

    #MAY NO LONGER BE NEEDED?
    if len(moonsets) == 0: #If no moonset found, add another day to search forward
        t1 = ts.from_astropy(obs_date+TimeDelta(2,format="jd"))
        f = almanac.risings_and_settings(eph, eph['Moon'], location)
        t, y = almanac.find_discrete(t0, t1, f)
        moonsets = t[y==moonrise]

    try:
        moonset = moonsets.utc_iso()[0]

    except:
        print(f"Error calculating moonset for Longitude: {lon} Latitude: {lat}.")
        moonset = obs_date

    return moonset

def get_sunset_time(obs_date,lat, lon,sunrise=False):
    #Gets sunset time using skyfield (MEDIUM)
    location = api.wgs84.latlon(lat,lon)

    t0 = ts.from_astropy(obs_date)
    t1 = ts.from_astropy(obs_date+TimeDelta(1,format="jd"))

    f = almanac.sunrise_sunset(eph, location)
    t, y = almanac.find_discrete(t0, t1, f)

    sunsets = t[y==sunrise]

    #MAY NO LONGER BE NEEDED?
    if len(sunsets) == 0: #If no moonset found, add another day to search forward
        t1 = ts.from_astropy(obs_date+TimeDelta(2,format="jd"))
        f = almanac.sunrise_sunset(eph, location)
        t, y = almanac.find_discrete(t0, t1, f)
        sunsets = t[y==sunrise]

    try:
        sunset = sunsets.utc_iso()[0]

    except:
        print(f"Error calculating sunset for Longitude: {lon} Latitude: {lat}.")
        sunset = obs_date

    return sunset

def get_sunset_moonset(d,coords,display=False): #NOT IN USE
    #Gets sunset and moonset using astroplan (VERY SLOW)

    obs = Observer(location=coords, timezone="UTC")

    moonset=obs.moon_set_time(time=d,which='next',n_grid_points=150)
    sunset=obs.sun_set_time(time=d,which='next',n_grid_points=150)

    LAG = (moonset.to_value("jd")-sunset.to_value("jd"))*24*60

    if display:
        print(f"MOONSET: {moonset}")
        print(f"SUNSET: {sunset}")
        print(f"LAG: {LAG:.5} mins") #Lag time

    return sunset, moonset

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
    #new_moon_date = new_moon.utc_datetime().replace(hour=0, minute=0,second=0)
    return Time(new_moon.utc_datetime())

def get_moon_age(obs_date):
    #Gets age of moon as a number of days
    last_new_moon = get_new_moon_date(obs_date)
    moon_age = obs_date-last_new_moon
    return moon_age.jd

#CALCULATE Q-VALUE
def get_moon_params(d,lat,lon,sunset=None,moonset=None,time_given=False,display=False,plot24hrs=False):
    #This calculates the q-test value and other moon params

    #Create coordinates object
    #Latitude is -90 to 90
    latitude = Latitude(lat*u.deg)
    #Longitude is -180 to +180
    longitude = Longitude(lon*u.deg,wrap_angle=180*u.deg)
    coords=EarthLocation.from_geodetic(lon=longitude,lat=latitude)

    
    #Get moon age
    MOON_AGE = get_moon_age(d)
    use_rises = MOON_AGE >=20
    
    #Calculate sunset and moonset time if not given
    if plot24hrs:
        sun_moonset_date = get_sun_moonset_date(d,lat,lon)
    else:
        sun_moonset_date = d
    if sunset is None:
        sunset = get_sunset_time(sun_moonset_date,lat,lon,use_rises)
    if moonset is None:
        moonset = get_moonset_time(sun_moonset_date,lat,lon,use_rises)

    #Other ways to calculate sun/moonset
    #sunset, moonset = get_sunset_moonset(d, coords)


    best_obs_time = get_best_obs_time(sunset, moonset)
    #Calculate best observation time if no time given
    if not time_given:
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

    if display:
        #Extra stuff

        #Calculate DAZ
        DAZ = sun_altaz.az - moon_altaz.az

        #Calculate moon semi-diameter and width
        W = SD*(1 - np.cos(ARCL.radian))

        #Calculate q-test value
        q = get_q_value(ARCV, W)

        
        LAG = (Time(moonset).to_value("jd")-Time(sunset).to_value("jd"))*24*60

        #Calculate illumination
        ILLUMINATION = 0.5*(1 - np.cos(ARCL.radian))

        #Cosine test: cos ARCL = cos ARCV cos DAZ
        cos_test = np.abs(np.cos(ARCL.radian)-np.cos(ARCV.radian)*np.cos(DAZ.radian))

        print(f"OBS LAT: {lat}. LON: {lon}")
        print(f"OBS TIME: {d.to_datetime().hour}:{d.to_datetime().minute}")
        print(f"BEST OBS TIME: {best_obs_time.to_datetime().hour}:{best_obs_time.to_datetime().minute}")
        print(f"DATE: {d.to_value('datetime')}")
        print(f"JULIAN DATE: {d.to_value('jd')}")
        print(f"MOON AGE: {round(MOON_AGE,3)}")
        print(f"MOONSET: {moonset}")
        print(f"SUNSET: {sunset}")
        
        print(f"LAG: {LAG:.5} mins") #Lag time

        print(f"MOON ALT: {moon_altaz.alt:.4}. AZ: {moon_altaz.az:.4}")
        print(f"SUN ALT: {sun_altaz.alt:.4}. AZ: {sun_altaz.az:.4}")
        print(f"EARTH-MOON DIST: {MOON_EARTH_DIST:.2}")
        print(f"SUN-MOON DIST: {DIST:.2}")

        print(f"ARCL: {ARCL:.3}")
        print(f"ARCV: {ARCV:.3}")
        print(f"DAZ: {DAZ:.3}")
        print(f"PARALLAX: {parallax.arcmin:.3} arcmin")
        print(f"ILLUMINATION: {ILLUMINATION:.3}")

        print(f"h: {h:.3}")
        print(f"W: {W.arcmin:.4} arcmin")
        print(f"W': {W_dash.arcmin:.4} arcmin")
        print(f"q(W): {q:.6}")
        print(f"q(W'): {q_dash:.6}")

        print(f"COS TEST: {cos_test:.4}")

        print()

    return q_dash

#PLOTTING MAP ----------------------------------------------------------------

def plot_visibility_at_date(obs_date, no_of_points=20):
    #Plots a visibility graph at a specified date

    #lat/long over the globe
    lat_arr = np.linspace(-60, 60, no_of_points)
    long_arr = np.linspace(-180, 180, no_of_points)
    q_vals = np.zeros((len(lat_arr),len(long_arr)))

    start = time.time()

    for i, latitude in enumerate(lat_arr):
        lap = time.time()
        print(f"Calculating latitude {round(lat_arr[i],2)} at time={round(lap-start,2)}s")

        #TEMP REMOVED ARRAY SUNSET FINDING
        #full_lat_arr = np.full(len(long_arr),latitude)
        #sunsets = get_sunset_time(obs_date, full_lat_arr, long_arr)
        #moonsets = get_moonset_time(obs_date, np.full(len(lat_arr),lat_arr[i]), long_arr)
        for j, longitude in enumerate(long_arr):
            #q_vals[j,i] = get_moon_params(obs_date, latitude, longitude, sunset=sunsets[j])

            q_vals[j,i] = get_moon_params(obs_date, latitude, longitude,plot24hrs=True)


    print(f"Total time: {round(time.time()-start,2)}s")
    print(f"Max q: {round(np.max(q_vals),3)}. Min q: {round(np.min(q_vals),3)}")
    create_contour_plot(obs_date, lat_arr,long_arr, q_vals)

    #create_globe_plot(obs_date, lat_arr,long_arr, q_vals)

    create_globe_plot_set(obs_date, lat_arr,long_arr, q_vals)

    #create_globe_animation(obs_date, lat_arr,long_arr, q_vals)

def create_globe_animation(obs_date, lat_arr,long_arr, q_val):
    #Plots moon visibility across a 3D globe
    #conda install -c conda-forge geovista

    x, y = np.meshgrid(long_arr,lat_arr,indexing='ij')
    mesh = gv.Transform.from_2d(x, y, data=q_val,radius=1.0)

    colors = [(1, 0, 0),(1,1,0), (0, 1, 0)]
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=6)

    plotter=gv.GeoPlotter()

    #Create colourmap
    yallop_annotations = {
    1: 'I(I)',
    2: 'I(V)',
    3: 'V(F)',
    4: 'V(V)',
    5: 'V',
    6: 'X'}

    sargs = dict(title="q-value",
                 interactive=True,
                 n_labels=6,
                 italic=True,
                 font_family="times",
                 title_font_size=22,
                 label_font_size=22)

    plotter.add_mesh(mesh, show_edges=False,
                     annotations = yallop_annotations,
                     clim=[-0.293,+0.216],
                     cmap=custom_cmap,
                     scalar_bar_args=sargs,
                     below_color= [1.0, 0.0, 0.0, 1.0],
                     above_color= [0.0, 1.0, 0.0, 1.0],
                     opacity=1)

    plotter.add_base_layer(texture=gv.natural_earth_1(),opacity=0.5)
    #plotter.add_coastlines(resolution="10m",opacity=1)

    plotter.view_xy()
    plotter.add_axes()
    title_date = obs_date.to_datetime().date()
    plotter.add_title(f"Global moon visibility at best time ({title_date})",font="arial",font_size=12)
    #plotter.export_obj(f"Globes\\{title_date} {len(long_arr)}x{len(long_arr)}.obj")
    plotter.show()

def create_globe_plot(obs_date,lat_arr,long_array,q_val):
    #Plots moon visibility across a globe

    #Set centre of globe
    PLOT_CENTRE = [0,0]

    x, y = np.meshgrid(long_array,lat_arr,indexing='ij')
    plt.figure(figsize=(9,5))
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'

    ax = plt.axes(projection=ccrs.Orthographic(*PLOT_CENTRE))

    ax.add_feature(cfeature.OCEAN, zorder=0)
    ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black')

    ax.set_global()
    ax.coastlines()
    #ax.world()

    crs = ccrs.PlateCarree()

    #custom colormap created, red to green 6 bins
    colors = [(1, 0, 0),(1,1,0), (0, 1, 0)]
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=6)

    cs = plt.contourf(x, y , q_val, levels = [-0.293,-0.232, -0.160, -0.014, +0.216],
                       alpha=0.6, cmap=custom_cmap ,extend='max',transform=crs)

    plt.colorbar(cs)
    nm = cs.legend_elements()[0]
    lbl = ['I(I)', 'I(V)', 'V(F)', 'V(V)', 'V']
    plt.legend(nm, lbl, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(lbl))

    title_date = obs_date.to_datetime().date()
    plt.title(f"Global moon visibility at best time ({title_date})")
    plt.show()

def create_globe_plot_set(obs_date,lat_arr,long_array,q_val):
    #Plots moon visibility across a globe at four views

    #Array of centers
    PLOT_CENTRE_ARR = [[0,0],[90,0],[180,0],[-90,0]]

    x, y = np.meshgrid(long_array,lat_arr,indexing='ij')
    fig = plt.figure(layout='constrained', figsize=(10, 8))

    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'

    colors = [(1, 0, 0),(1,1,0), (0, 1, 0)]
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=6)

    for i in range(len(PLOT_CENTRE_ARR)):
        ax = fig.add_subplot(2,2,i+1,projection=ccrs.Orthographic(*PLOT_CENTRE_ARR[i]))

        ax.add_feature(cfeature.OCEAN, zorder=0)
        ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black')
        ax.set_global()
        crs = ccrs.PlateCarree()

        # plot discontinuity line between 180,-90 to 180,90
        ax.plot([180, -90], [180, 90], linestyle='dashed',transform=crs,
                linewidth=5) #doesnt work

        cs = ax.contourf(x, y , q_val, levels = [-0.293,-0.232, -0.160, -0.014, +0.216],
                      alpha=0.6, cmap=custom_cmap ,extend='max',transform=crs)


    fig.colorbar(cs, ax=fig.axes, shrink=0.9)
    nm = cs.legend_elements()[0]
    lbl = ['I(I)', 'I(V)', 'V(F)', 'V(V)', 'V']
    fig.legend(nm, lbl, loc='upper center', bbox_to_anchor=(0.5, -0.05)
               , ncol=len(lbl), fontsize=17)

    title_date = obs_date.to_datetime().date()
    plt.suptitle(f"Global moon visibility at best time ({title_date})", fontsize=25)
    plt.savefig(f"Global moon visibility at best time globes ({title_date}).png",dpi=200)
    plt.show()

def create_contour_plot(obs_date,lat_arr,long_array,q_val):
    #Plots moon visibility across a world map
    x, y = np.meshgrid(long_array,lat_arr,indexing='ij')
    plt.figure(figsize=(9,5))
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'

    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    #ax.world()

    #custom colormap created, red to green 6 bins
    colors = [(1, 0, 0),(1,1,0), (0, 1, 0)]
    custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=6)

    cs = plt.contourf(x, y , q_val, levels = [-0.293,-0.232, -0.160, -0.014, +0.216],
                       alpha=0.6, cmap=custom_cmap ,extend='max')

    #cs = plt.contourf(x, y , q_val,alpha=0.6, cmap=custom_cmap ,extend='max')

    plt.colorbar(cs)
    nm = cs.legend_elements()[0]
    lbl = ['I(I)', 'I(V)', 'V(F)', 'V(V)', 'V']
    plt.legend(nm, lbl, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(lbl))

    plt.ylim(-90,90)
    plt.xlim(-180,180)

    title_date = obs_date.to_datetime().date()
    plt.title(f"Global moon visibility at best time ({title_date})")
    plt.savefig(f"Global moon visibility at best time ({title_date}).png",dpi=200)
    plt.show()

date_to_plot = Time("2023-03-22")
plot_visibility_at_date(date_to_plot,40)

date_to_check = Time("1989-4-4") #Works, best time is 22:00 1990-11-19
lat = 41.9
lon = -88.7
#get_moon_params(date_to_check,lat,lon,display=True)
