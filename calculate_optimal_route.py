import pandas as pd
import geopandas as gpd
import openrouteservice as ors
from openrouteservice import exceptions
import re
from openrouteservice.optimization import Vehicle, Job, Shipment, ShipmentStep
from shapely.geometry import Point, LineString
import os
from dotenv import load_dotenv

# Set OpenRouteService API key
load_dotenv()
ors_api_key = os.environ.get('OPENROUTESERVICE_KEY')
if not ors_api_key:
    raise ValueError("OPENROUTESERVICE_KEY environment variable not set.")
ors_client = ors.Client(key=ors_api_key)

def optimal_route(input_source, input_destinations, input_dest_id_field, input_final_stop, 
                   bridges_gdf=None, priority_rule=None):  
    """
    Calculate optimal route with optional constraints
    
    Parameters:
    -----------
    bridges_gdf : GeoDataFrame, optional
        GeoDataFrame of bridges with 'status' column (operational/non-operational)
    priority_rule : dict, optional
        Dictionary with 'before_dest' and 'after_dest' keys specifying ordering constraint
        Example: {'before_dest': 'Site A', 'after_dest': 'Site B'} means Site A must come before Site B
    """
    # Clean column names and prepare data
    source = input_source.copy()
    source['name'] = "Starting point"
    final_stop = input_final_stop.copy()
    final_stop['name'] = "Final stop"
    destinations = input_destinations.copy()

    # Harmonize coordinate column names
    for df in [source, final_stop, destinations]:
        df.columns = df.columns.str.lower()
        df.rename(columns={'latitude': 'lat', 'longitude': 'lon', 'y': 'lat', 'x': 'lon'}, inplace=True)

    # Convert to GeoDataFrames
    source = gpd.GeoDataFrame(source, geometry=gpd.points_from_xy(source.lon, source.lat), crs="EPSG:4326")
    final_stop = gpd.GeoDataFrame(final_stop, geometry=gpd.points_from_xy(final_stop.lon, final_stop.lat), crs="EPSG:4326")
    destinations = gpd.GeoDataFrame(destinations, geometry=gpd.points_from_xy(destinations.lon, destinations.lat), crs="EPSG:4326")

    # Filter out destinations near non-operational bridges if bridges data is provided
    if bridges_gdf is not None and len(bridges_gdf) > 0:
        # Make sure status column exists and filter non-operational
        if 'status' in bridges_gdf.columns:
            # Convert status to string and then check for non-operational
            bridges_gdf_copy = bridges_gdf.copy()
            bridges_gdf_copy['status'] = bridges_gdf_copy['status'].astype(str)
            non_operational = bridges_gdf_copy[bridges_gdf_copy['status'].str.lower() == 'non-operational']
            
            if len(non_operational) > 0:
                # Buffer non-operational bridges by 100m to create exclusion zones
                non_operational_buffered = non_operational.to_crs(non_operational.estimate_utm_crs())
                non_operational_buffered['geometry'] = non_operational_buffered.geometry.buffer(100)
                non_operational_buffered = non_operational_buffered.to_crs('EPSG:4326')
                
                # Check which destinations are near non-operational bridges
                destinations_check = destinations.sjoin(non_operational_buffered, how='left', predicate='intersects')
                excluded = destinations_check[destinations_check.index_right.notna()]
                
                if len(excluded) > 0:
                    excluded_names = excluded[input_dest_id_field].tolist()
                    raise ValueError(
                        f"The following destinations are near non-operational bridges and cannot be included: {', '.join(map(str, excluded_names))}. "
                        "Please remove them or wait until the bridges are operational."
                    )

    # Create vehicle and jobs objects
    home_base = source.geometry.iloc[0].coords[0]
    final_dest = final_stop.geometry.iloc[0].coords[0]
    stops = destinations.geometry.apply(lambda geom: geom.coords[0]).tolist()

    priority_positions = None
    
    if priority_rule and 'before_dest' in priority_rule and 'after_dest' in priority_rule:
        before_dest_name = priority_rule['before_dest']
        after_dest_name = priority_rule['after_dest']

        if 'before_idx' in priority_rule and 'after_idx' in priority_rule:
            before_pos = int(priority_rule['before_idx'])
            after_pos = int(priority_rule['after_idx'])
        else:
            before_matches = destinations[destinations[input_dest_id_field] == before_dest_name]
            after_matches = destinations[destinations[input_dest_id_field] == after_dest_name]

            if len(before_matches) == 0 or len(after_matches) == 0:
                before_pos = after_pos = None
            else:
                before_pos = destinations.index.get_loc(before_matches.index[0])
                after_pos = destinations.index.get_loc(after_matches.index[0])

        if before_pos is not None and after_pos is not None:
            if before_pos == after_pos:
                raise ValueError("Priority rule must use two different destinations.")
            if not (0 <= before_pos < len(stops)) or not (0 <= after_pos < len(stops)):
                raise ValueError("Priority rule destinations are no longer available.")
            priority_positions = (before_pos, after_pos)

    vehicle = Vehicle(id=1, profile="driving-hgv", start=home_base, end=final_dest)
    
    # Create jobs. A visit-order rule is modeled as a shipment, which enforces
    # pickup-before-delivery instead of the soft preference from job priority.
    jobs = []
    shipments = []
    for i, loc in enumerate(stops):
        if priority_positions and i in priority_positions:
            continue
        jobs.append(Job(id=i+1, location=loc))

    if priority_positions:
        before_pos, after_pos = priority_positions
        shipments.append(
            Shipment(
                pickup=ShipmentStep(id=before_pos+1, location=stops[before_pos]),
                delivery=ShipmentStep(id=after_pos+1, location=stops[after_pos]),
                amount=[],
                skills=[],
                priority=0
            )
        )

    # Create a lookup list of all points before the API call
    all_points_for_lookup = []
    source_name = input_source.get('name', pd.Series(['Starting point'])).iloc[0]
    all_points_for_lookup.append({'name': source_name, 'lon': source.geometry.x.iloc[0], 'lat': source.geometry.y.iloc[0]})
    
    final_stop_name = input_final_stop.get('name', pd.Series(['Final stop'])).iloc[0]
    all_points_for_lookup.append({'name': final_stop_name, 'lon': final_stop.geometry.x.iloc[0], 'lat': final_stop.geometry.y.iloc[0]})

    for _, row in destinations.iterrows():
        point_name = row[input_dest_id_field] if input_dest_id_field in row else "Unnamed Destination"
        all_points_for_lookup.append({'name': point_name, 'lon': row.geometry.x, 'lat': row.geometry.y})
    
    try:
        # Get the optimized itinerary
        opt = ors_client.optimization(jobs=jobs, vehicles=[vehicle], shipments=shipments, geometry=True)

    # Handle various error types given the returned error messages from the API
    except exceptions.ApiError as e:
        error_message = str(e)
        lon_err, lat_err = None, None

        # Handle "Could not find routable point" error
        if 'Could not find routable point' in error_message:
            match = re.search(r"coordinate \d+: (-?\d+\.?\d*)\s+(-?\d+\.?\d*)", error_message)
            if match:
                lon_err, lat_err = float(match.group(1)), float(match.group(2))

        # Handle "Unfound route(s) from location" for start/end/multiple destinations
        elif 'Unfound route(s) from location' in error_message:
            # This error uses [lon,lat] format
            match = re.search(r"location \[(-?\d+\.?\d*),(-?\d+\.?\d*)\]", error_message)
            if match:
                lon_err, lat_err = float(match.group(1)), float(match.group(2))

        # If we successfully parsed coordinates from any known error format, find the point name
        if lon_err is not None and lat_err is not None:
            for point in all_points_for_lookup:
                if abs(point['lon'] - lon_err) < 0.0001 and abs(point['lat'] - lat_err) < 0.0001:
                    # Raise a new, more descriptive error for the Shiny app to display
                    raise ValueError(
                        f"Unroutable Location: The point named '{point['name']}' could not be reached. "
                        "Places that are more than 500m from a road are considered to be unreachable!"
                    )
        
        # If the error message is not handled under the two cases above or parsing failed, re-raise the original error
        raise e

    steps = opt['routes'][0]['steps'] 
    
    job_sequence = pd.DataFrame([
        {'stop_id': item.get('job') or item.get('pickup') or item.get('delivery'),
         'distance': item.get('distance'),
         'location': item['location']} 
        for item in steps 
        if item.get('type') in ['job', 'pickup', 'delivery'] and item.get('location') is not None
    ])
    job_sequence = job_sequence.sort_values( by='distance') 
    job_sequence = job_sequence.reset_index(drop=True)
    job_sequence['rank'] = job_sequence.index + 1

    job_sequence['lon'] = job_sequence['location'].apply(lambda loc: loc[0])
    job_sequence['lat'] = job_sequence['location'].apply(lambda loc: loc[1])

    gdf = gpd.GeoDataFrame(
        job_sequence, 
        geometry=[Point(xy) for xy in zip(job_sequence['lon'], job_sequence['lat'])],
        crs="EPSG:4326"
    )
    
    gdf = gdf.drop(columns=['location', 'lat', 'lon'])

    gdf = gdf.to_crs(gdf.estimate_utm_crs()) 
    gdf['geometry'] = gdf['geometry'].buffer(10)
    gdf = gdf.to_crs('EPSG:4326')

    destinations = gpd.sjoin(destinations, gdf, how="left", predicate="intersects")
    destinations['route_order'] = 'Destination ' + destinations['rank'].astype(str)
    cols = ['route_order'] + [col for col in destinations if col != 'route_order']
    destinations = destinations[cols]
    destinations = destinations.sort_values(by='rank')
    destination_columns = ['route_order', input_dest_id_field, 'distance', 'geometry']
    destinations = destinations.loc[:, ~destinations.columns.duplicated()]
    destinations = destinations[list(dict.fromkeys(destination_columns))]

    locations = pd.DataFrame(steps, columns=['type', 'job', 'pickup', 'delivery', 'location', 'distance'])
    locations = locations.sort_values( by='distance')
    locations = locations.reset_index(drop=True)

    locations[['lon', 'lat']] = pd.DataFrame(locations['location'].tolist(), index=locations.index)
    locations.rename(columns={'type': 'name'}, inplace=True)
    service_stop_count = 0

    def label_route_step(row):
        nonlocal service_stop_count
        if row['name'] in ['job', 'pickup', 'delivery']:
            service_stop_count += 1
            return f'destination {service_stop_count}'
        if row['name'] == 'start':
            return 'home_base'
        return 'final_stop'

    locations['name'] = locations.apply(lambda row: 
                                        label_route_step(row), axis=1)
    ordered_coords = locations['location'].tolist()
    locations.drop(columns=['job', 'pickup', 'delivery', 'location'], inplace=True)

    route_segments = pd.DataFrame({
        'segment_name': locations['name'].shift(1, fill_value='home_base') + ' to ' + locations['name'],
        'origin_lon': locations['lon'].shift(1),
        'origin_lat': locations['lat'].shift(1),
        'end_lon': locations['lon'],
        'end_lat': locations['lat'],
        'distance': locations['distance']
    })

    route_segments['distance'] = route_segments['distance'].diff()
    route_segments['distance'] = (route_segments['distance']/1000).round(2)
    route_segments = route_segments.iloc[1:].reset_index(drop=True)

    destination_labels = destinations['route_order'].astype(str).str.lower()
    destination_ids = destinations[input_dest_id_field].astype(str)
    dest_names = dict(zip(destination_labels, destination_ids))

    def replace_names(text, name_list):
        # Sort by key length descending to replace longer strings first
        # This prevents substring matching issues (e.g., "destination 1" matching in "destination 10")
        sorted_items = sorted(name_list.items(), key=lambda x: len(x[0]), reverse=True)
        for key, value in sorted_items:
            text = text.replace(key, str(value))  # Convert value to string to handle any data type
        return text

    route_segments['segment_name'] = route_segments['segment_name'].apply(lambda x: replace_names(x, dest_names))

    # Enhanced API call: Request both surface AND road class information
    directions = ors_client.directions(
        coordinates=ordered_coords,
        profile='driving-car',
        extra_info=['surface', 'roadaccessrestrictions', 'waycategory'],
        format='geojson'
    )

    surface_details = directions['features'][0]['properties']['extras']['surface']['values']
    surface_summary = directions['features'][0]['properties']['extras']['surface']['summary']
    
    # Get way category (road class) information
    waycategory_details = directions['features'][0]['properties']['extras'].get('waycategory', {}).get('values', [])
    
    route_geom = directions['features'][0]['geometry']
    
    def add_road_attributes(route_geometry, surface_details, waycategory_details):
        coords = route_geometry['coordinates']
        route = LineString(coords)
        gdf = gpd.GeoDataFrame(geometry=[route], crs='EPSG:4326')
        
        def split_line(line):
            return [LineString([line.coords[i], line.coords[i+1]]) 
                    for i in range(len(line.coords) - 1)]
        
        gdf['segments'] = gdf.geometry.apply(split_line)
        
        gdf = gdf.explode('segments')
        gdf = gdf.reset_index(drop=True)
        gdf['geometry'] = gdf['segments']
        gdf = gdf.drop(columns=['segments'])
        
        # Add surface information
        gdf['surface'] = None
        for start, end, surface_id in surface_details:
            gdf.loc[start:end, 'surface'] = surface_id
        
        # Add way category (road class) information
        gdf['road_class'] = None
        for start, end, class_id in waycategory_details:
            gdf.loc[start:end, 'road_class'] = class_id
        
        return gdf

    route_detailed = add_road_attributes(route_geom, surface_details, waycategory_details)

    # Surface code mappings
    surface_codes = {
        0: "Unknown", 1: "Paved", 2: "Unpaved", 3: "Asphalt", 4: "Concrete",
        6: "Metal", 7: "Wood", 8: "Compacted Gravel", 10: "Gravel", 11: "Dirt",
        12: "Ground", 13: "Ice", 14: "Paving Stones", 15: "Sand", 17: "Grass",
        18: "Grass Paver"
    }
    
    # Road class codes (way category)
    road_class_codes = {
        0: "Highway/Motorway",
        1: "National Road",
        2: "Regional Road", 
        3: "Local Road",
        4: "Urban Road",
        5: "Access Road",
        6: "Track"
    }
    
    # Map surface and road class codes
    route_detailed['surface_raw'] = route_detailed['surface'].map(lambda x: surface_codes.get(x, "Unknown"))
    route_detailed['road_class_name'] = route_detailed['road_class'].map(lambda x: road_class_codes.get(x, "Unknown"))
    
    # Classification logic: Determine if road is Paved or Unpaved
    def classify_pavement(row):
        surface = row['surface_raw']
        road_class = row['road_class']
        
        # Known paved surfaces
        if surface in ['Paved', 'Asphalt', 'Concrete', 'Paving Stones', 'Metal']:
            return 'Paved'
        
        # Known unpaved surfaces
        if surface in ['Unpaved', 'Gravel', 'Compacted Gravel', 'Dirt', 'Ground', 'Sand', 'Grass', 'Grass Paver', 'Wood']:
            return 'Unpaved'
        
        # For Unknown surfaces, use road class as a proxy
        if surface == 'Unknown' or surface == 'Ice':
            if road_class is not None:
                # Highway, National, Regional roads are typically paved
                if road_class in [0, 1, 2]:
                    return 'Paved (inferred from road class)'
                # Urban roads are usually paved
                elif road_class == 4:
                    return 'Paved (inferred from road class)'
                # Local roads could go either way, default to Unknown
                elif road_class == 3:
                    return 'Unknown'
                # Access roads and tracks are often unpaved
                elif road_class in [5, 6]:
                    return 'Unpaved (inferred from road class)'
            
            # If no road class information, remain Unknown
            return 'Unknown'
        
        return 'Unknown'
    
    route_detailed['pavement_status'] = route_detailed.apply(classify_pavement, axis=1)
    
    # Create simplified pavement category (just Paved/Unpaved/Unknown)
    def simplify_pavement(status):
        if 'Paved' in status:
            return 'Paved'
        elif 'Unpaved' in status:
            return 'Unpaved'
        else:
            return 'Unknown'
    
    route_detailed['surface'] = route_detailed['pavement_status'].apply(simplify_pavement)

    route_detailed = route_detailed.to_crs(route_detailed.estimate_utm_crs())
    route_detailed['segment_length'] = route_detailed.geometry.length
    
    cols = [col for col in route_detailed if col != 'geometry'] + ['geometry']
    route_detailed = route_detailed[cols]
    route_detailed = route_detailed.to_crs('EPSG:4326')

    return route_detailed, source, final_stop, destinations, route_segments, input_dest_id_field
