'''
Folium map with Volcanoes in USA and World Population.
Layer control also added.

Used list comprehension over pandas dataframe.
'''

import os
import time

import folium
from selenium import webdriver


def colour_producer(elev):
    if elev < 1200:
        return 'green'
    elif 1000 <= elev < 3000:
        return 'orange'
    else:
        return 'red'


map = folium.Map(
    location=[38.58, -115.0],
    zoom_start=6,
    tiles="Mapbox Bright"
)

fgv = folium.FeatureGroup(
    name='Volcanoes'
)

# for + list
# coordinates = []
# with open('./Volcanoes_USA.txt','r') as file:
#     for line in file:
#         coordinates.append(line.strip('\n').split(',')[8:])

# list comprehension
coordinates = [line.rstrip('\n').split(',') for line in open('./Volcanoes_USA.txt', 'r')]

# remove header
coordinates = coordinates[1:]

# print(coordinates)

for i, coord in enumerate(coordinates):
    fgv.add_child(folium.CircleMarker(
        location=[float(coord[8]), float(coord[9])],
        popup='Elevation: {} m, Marker: #{}'.format(round(float(coord[5]), 2), str(i + 1)),
        fill_color=colour_producer(float(coord[5])),
        radius=6,
        color='grey',
        fill_opacity=0.7
    ))

fgp = folium.FeatureGroup(
    name='Population'
)

fgp.add_child(folium.GeoJson(
    data=(open('./world.json', 'r', encoding='utf-8-sig').read()),
    style_function=lambda x: {
        'fillColor': 'green' if x['properties']['POP2005'] < 10000000 else 'orange' if 10000000 <= x['properties'][
            'POP2005'] < 20000000 else 'red'}
))

map.add_child(fgv)
map.add_child(fgp)
map.add_child(folium.LayerControl())

delay = 5
fn = 'Map1.html'
tmpurl = 'file://{path}/{mapfile}'.format(path=os.getcwd(), mapfile=fn)
map.save(fn)

browser = webdriver.Chrome()
browser.get(tmpurl)
# Give the map tiles some time to load
time.sleep(delay)
browser.save_screenshot('Map1.png')
browser.quit()
