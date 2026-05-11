import time
import numpy as np
import pandas as pd
from geopy.geocoders import Nominatim
from sklearn.cluster import DBSCAN

class GeoAnalyzer:
    def __init__(self):
        self.geolocator = Nominatim(user_agent="my_analyzer")

    def get_coords_list(self, places_counter):
        results = []
        for name, count in places_counter.most_common(100):
            try:
                location = self.geolocator.geocode(name, language="ru", timeout=5)
                if location:
                    results.append({
                        "name": name,
                        "lat": location.latitude,
                        "lon": location.longitude,
                        "weight": count,
                        "address": location.address
                    })
                time.sleep(1.1)
            except:
                continue
        return pd.DataFrame(results)

    def predict_location(self, df_coords):
        if df_coords.empty:
            return None, None

        coords = df_coords[['lat', 'lon']].values
        kms_per_radian = 6371.0088
        epsilon = 40 / kms_per_radian  # Радиус 40 км

        db = DBSCAN(eps=epsilon, min_samples=1, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
        df_coords['cluster'] = db.labels_

        cluster_weights = df_coords.groupby('cluster')['weight'].sum()
        best_cluster_id = cluster_weights.idxmax()
        best_cluster_points = df_coords[df_coords['cluster'] == best_cluster_id]
        main_point = best_cluster_points.sort_values(by='weight', ascending=False).iloc[0]

        final_info = self.geolocator.reverse(f"{main_point['lat']}, {main_point['lon']}", language="ru")
        address = final_info.raw.get('address', {})

        city = address.get('city') or address.get('town') or address.get('village') or address.get('state')
        country = address.get('country')

        return city, country
