import gpsd
import reverse_geocoder as rg
from arcgis.gis import GIS
import pickle

# Connect to the local gpsd
gpsd.connect()

# Connect somewhere else
# gpsd.connect(host="127.0.0.1", port=123456)

# Get gps position
packet = gpsd.get_current()

# See the inline docs for GpsResponse for the available data
gis = GIS()

while True:
    print("-")
    packet = gpsd.get_current()
    # See the inline docs for GpsResponse for the available data
    try:
        coordinates = packet.position()
        print(coordinates)
        result = rg.search(coordinates)
        with open('curr_gps_coords.pkl', 'wb') as fh:
            pickle.dump(coordinates, fh)
    except:
        print("gps error")
