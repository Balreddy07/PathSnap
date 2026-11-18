import sys, os
import pandas as pd
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(repo_root)

from src.data_loader import load_trip
from src.preprocessing.gps_preprocessing import preprocess_gps_trajectory
from src.preprocessing.road_preprocessing import preprocess_road_network
from src.utils.spatial_index import RoadNetworkSpatialIndex
from src.preprocessing.candidate_generation import generate_trip_candidates
from src.preprocessing.label_generation import generate_labels

trip_id = '00000001'
trip = load_trip(trip_id, partition='train')
track = trip['track']
nodes = trip['nodes']
arcs = trip['arcs']
route = trip['route']

# GPS preprocessing
gps_df = preprocess_gps_trajectory(track, trip_id)
print('GPS rows:', len(gps_df))

# Road preprocessing
arc_df, G = preprocess_road_network(nodes, arcs)
print('Arcs rows:', len(arc_df))
print('Graph edges:', G.number_of_edges(), 'nodes:', G.number_of_nodes())
if G.number_of_edges() != len(arc_df):
    print('WARNING: edge count differs from arc rows')
else:
    print('Duplicate arcs preserved')

# Spatial index
spatial_idx = RoadNetworkSpatialIndex(arc_df, nodes)
print('Spatial index built')

# Candidate generation
candidates = generate_trip_candidates(gps_df, spatial_idx, trip_id)
print('Candidates generated:', len(candidates))
print('Unique (trip,gps) combos:', candidates.groupby(['trip_id','gps_index']).size().shape)

# Label generation
arc_geoms = {int(row.Index): (row.from_lon, row.from_lat, row.to_lon, row.to_lat) for row in arc_df.itertuples()}
route_df = pd.DataFrame({'trip_id':trip_id, 'arc_id':route.values})
labels_df = generate_labels(candidates, route_df, arc_geoms)
print('Labels attached, unique label values:', labels_df['label'].nunique())
print(labels_df.head().to_dict(orient='records'))
