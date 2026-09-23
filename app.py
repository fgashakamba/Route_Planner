import shiny
from shiny import App, render, ui, reactive
from shiny.types import SilentException
import pandas as pd
import geopandas as gpd
import openrouteservice as ors
import folium
from folium import MacroElement
from jinja2 import Template
import os
import time
import tempfile
import urllib.parse
import asyncio
from calculate_optimal_route import optimal_route

# Load auxiliary layers
lakes = gpd.read_file(os.path.join(os.path.dirname(__file__),  'data_wgs84', 'RW_lakes.gpkg'))
np_parks = gpd.read_file(os.path.join(os.path.dirname(__file__), 'data_wgs84', 'RW_national_parks.gpkg'))
country = gpd.read_file(os.path.join(os.path.dirname(__file__), 'data_wgs84', 'RW_country.gpkg'))

# Load bridges data
try:
    bridges = gpd.read_file(os.path.join(os.path.dirname(__file__), 'data_wgs84', 'RW_bridges.gpkg'))
    bridges.columns = bridges.columns.str.lower()
    if 'status' not in bridges.columns:
        raise ValueError("Bridges file must contain 'status' column")
    bridges_available = True
except (FileNotFoundError, Exception) as e:
    print(f"Warning: Could not load bridges data: {e}")
    bridges_available = False
    bridges = None

# Load pre-defined destinations database
try:
    destinations_db = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'destinations_database.csv'))
    destinations_db.columns = destinations_db.columns.str.lower().str.replace(' ', '_')
    required_cols = ['name', 'latitude', 'longitude']
    if not all(col in destinations_db.columns for col in required_cols):
        raise ValueError(f"Destinations database must contain columns: {required_cols}")
    destinations_available = True
    destination_choices = {str(idx): row['name'] for idx, row in destinations_db.iterrows()}
    
    if 'category' in destinations_db.columns:
        categories = destinations_db['category'].unique()
        category_choices = {cat: cat for cat in sorted(categories)}
    else:
        categories = None
        category_choices = {}

except (FileNotFoundError, Exception) as e:
    print(f"Warning: Could not load destinations database: {e}")
    destinations_available = False
    destination_choices = {}
    categories = None
    category_choices = {}

# Get the centroid of the country
centroid = country.to_crs(country.estimate_utm_crs()).geometry.centroid.to_crs('EPSG:4326').iloc[0]
center_coords = [centroid.y, centroid.x]

# UI
app_ui = ui.page_fluid(
    ui.tags.style(
        """
        .shiny-input-container {
            height: 60px;
            padding: 10px 10px 60px 10px;
        }
        .shiny-input-container > label {
            font-size: 16px;
        }
        .destination-selector {
            max-height: 300px;
            overflow-y: auto;
            border: 1px solid #ddd;
            padding: 10px;
            margin: 5px 0;
        }
        .selected-destinations {
            background-color: #f8f9fa;
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
        }
        .button-pills {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 15px;
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 8px;
        }
        .constraint-section {
            background-color: #f0f8ff;
            padding: 12px;
            margin: 10px 0;
            border-radius: 5px;
            border: 1px solid #cce7ff;
        }
        """
    ),
    ui.panel_title("Tubura Route Optimization Tool, Enhanced Prototype v4 (January, 2026)"),
    ui.layout_columns(
        ui.row(
            ui.column(3,
                # Input method selection card
                ui.card(
                    ui.card_header("Choose Input Method"),
                    ui.p(
                        ui.input_radio_buttons(
                            id="input_method",
                            label="",
                            choices={
                                "select": "Select from database" if destinations_available else "Select from database (Not available)",
                                "upload": "Upload CSV files",
                                "map_click": "Click on map"
                            },
                            selected="select" if destinations_available else "upload"
                        )
                    )
                ),

                # Route configuration card
                ui.card(
                    ui.card_header("Route Configuration"),
                    ui.p(
                        ui.output_ui("input_controls")
                    )
                ),

                # Destination selection panel
                ui.output_ui("destination_selection_panel"),

                # Constraints card (NEW)
                ui.card(
                    ui.card_header("Route Constraints (Optional)"),
                    ui.div(
                        # Bridge constraint
                        ui.input_checkbox(
                            "use_bridge_constraint",
                            "Avoid non-operational bridges" if bridges_available else "Avoid non-operational bridges (No bridge data)",
                            value=bridges_available
                        ),
                        # Priority rule
                        ui.input_checkbox(
                            "use_priority_rule",
                            "Apply visit order rule (X before Y)",
                            value=False
                        ),
                        ui.output_ui("priority_rule_ui"),
                        class_="constraint-section"
                    )
                ),

                # Calculate button card
                ui.card(
                    ui.input_action_button("processButton", label="Calculate Optimal Route", class_="btn-primary w-100 mb-2")
                )
            ),
            ui.column(9,
                # Button pills at the top of the map
                ui.div(
                    ui.output_ui("google_maps_button"),  # NEW: Dynamic Google Maps button
                    ui.download_button(id="downloadRoute", label="Get route file", class_="btn-sm btn-secondary"),
                    ui.download_button(id="downloadRouteSegments", label="Get route segments distances", class_="btn-sm btn-secondary"),
                    ui.input_action_button("Show_Segments_Table", label="Show route segments", class_="btn-sm btn-info"),
                    ui.input_action_button("Show_Table", label="Show road surface statistics", class_="btn-sm btn-info"),
                    class_="button-pills"
                ),
                ui.card(
                    ui.p(
                        ui.output_ui("map")
                    ),
                    ui.card_footer("Paved roads are shown in Green, Unpaved in Orange, and Unknown in Maroon. Hover to see destination sequence.")
                )
            )
        )
    )
)

def server(input, output, session):
    def get_optional_input(input_id, default=None):
        try:
            return getattr(input, input_id)()
        except (AttributeError, SilentException):
            return default

    # Reactive values
    uploaded_files = reactive.Value({
        'source': None,
        'final_stop': None,
        'destinations': None
    })
    
    selected_destinations = reactive.Value([])
    dest_id_field = reactive.Value(None)
    dest_id_choices = reactive.Value([])
    result = reactive.Value(None)
    
    map_click_mode = reactive.Value("none")
    map_clicked_points = reactive.Value({'source': None, 'final_stop': None, 'destinations': []})
    rerender_trigger = reactive.Value(0)

    @reactive.extended_task
    async def calculate_route_task(source_df, destinations_df, id_field, final_stop_df, bridges_data, priority_rule):
        return await asyncio.to_thread(
            optimal_route,
            source_df,
            destinations_df,
            id_field,
            final_stop_df,
            bridges_gdf=bridges_data,
            priority_rule=priority_rule
        )

    @reactive.Effect
    def _():
        status = calculate_route_task.status()

        if status == "success":
            route_detailed, source, final_stop, destinations_result, route_segments, used_id = calculate_route_task.result()
            result.set({
                'route_detailed': route_detailed,
                'source': source,
                'final_stop': final_stop,
                'destinations': destinations_result,
                'route_segments': route_segments,
                'used_id_field': used_id
            })
            ui.notification_show("Route calculated successfully!", type="message", duration=3)
        elif status == "error":
            try:
                calculate_route_task.result()
            except ValueError as e:
                ui.notification_show(str(e), type="error", duration=10)
            except Exception as e:
                ui.notification_show(f"Error calculating route: {str(e)}", type="error", duration=10)
                import traceback
                print("".join(traceback.format_exception(type(e), e, e.__traceback__)))

    # Priority rule UI (NEW)
    @render.ui
    def priority_rule_ui():
        if not input.use_priority_rule():
            return None
        
        method = input.input_method()
        dest_names = []
        
        if method == "select" and destinations_available:
            selected = selected_destinations()
            if selected:
                dest_names = [destinations_db.loc[int(idx), 'name'] for idx in selected]
        elif method == "upload":
            files = uploaded_files()
            if files.get('destinations') is not None:
                try:
                    dest_df = pd.read_csv(files['destinations'])
                    id_field = dest_id_field()
                    if id_field and id_field in dest_df.columns:
                        dest_names = dest_df[id_field].tolist()
                except:
                    pass
        elif method == "map_click":
            points = map_clicked_points()
            dest_names = [dest['name'] for dest in points['destinations']]
        
        if not dest_names:
            return ui.p("Add destinations first to set priority rules", style="font-size: 12px; color: #666;")
        
        return ui.div(
            ui.input_select(
                "priority_before",
                "Visit this destination first:",
                choices={"": "Select..."} | {str(i): name for i, name in enumerate(dest_names)},
                selected=""
            ),
            ui.input_select(
                "priority_after",
                "Before visiting:",
                choices={"": "Select..."} | {str(i): name for i, name in enumerate(dest_names)},
                selected=""
            ),
            style="margin-top: 10px; padding: 10px; background: white; border-radius: 5px;"
        )

    # Render conditional input controls
    @render.ui
    def input_controls():
        method = input.input_method()

        if method == "upload":
            return ui.div(
                ui.input_file(id="source", label="Starting point coordinates:"),
                ui.input_file(id="final_stop", label="Final point coordinates:"),
                ui.input_file(id="destinations", label="Destinations coordinates:"),
                ui.input_select(id="dest_id_field", label="Destination ID field", choices=["Select field..."]),
            )
        elif method == "map_click":
            return ui.div(
                ui.p("Click on the map to set points:", style="font-weight: bold; color: #2c5282;"),
                ui.div(
                    ui.input_action_button("set_source_mode", "Set Starting Point", class_="btn-sm btn-success me-2"),
                    ui.input_action_button("set_final_mode", "Set Final Stop", class_="btn-sm btn-warning me-2"),
                    ui.input_action_button("set_dest_mode", "Add Destinations", class_="btn-sm btn-info me-2"),
                    ui.input_action_button("clear_map_points", "Clear All Points", class_="btn-sm btn-outline-danger"),
                    style="margin-bottom: 10px;"
                ),
                ui.output_ui("click_mode_status"),
                ui.output_ui("map_points_summary")
            )
        else:  # select method
            if not destinations_available:
                return ui.div(
                    ui.p("Destinations database is not available. Please use the upload method.",
                         style="color: #e74c3c; font-weight: bold;")
                )
            return ui.div(
                ui.input_select(
                    id="source_select",
                    label="Starting point:",
                    choices={"": "Select starting point", **destination_choices}
                ),
                ui.input_select(
                    id="final_stop_select",
                    label="Final stop:",
                    choices={"": "Select final stop", **destination_choices}
                ),
                # Category filter if available
                ui.input_select(
                    id="category_filter",
                    label="Filter destinations by category:",
                    choices={"": "All categories", **category_choices}
                ) if categories is not None else ui.div(),
            )

    # Destination selection panel for database method (RESTORED)
    @render.ui
    def destination_selection_panel():
        method = input.input_method()

        if method == "select" and destinations_available:
            return ui.card(
                ui.card_header("Select Destinations"),
                ui.p(
                    ui.input_text(
                        id="destination_search",
                        label="Search destinations:",
                        placeholder="Type to search..."
                    ),
                    ui.output_ui("destination_checkboxes"),
                    ui.br(),
                    ui.div(
                        ui.input_action_button("clear_selections", "Clear All Selections", class_="btn-sm btn-outline-secondary"),
                        style="margin-bottom: 10px;"
                    ),
                    ui.output_ui("selected_destinations_display")
                )
            )
        else:
            return ui.div()

    # Track selections across category changes (RESTORED)
    @reactive.Effect
    @reactive.event(input.selected_dest_checkboxes)
    def _():
        current_selections = input.selected_dest_checkboxes() or []
        if current_selections is not None:
            persistent_selections = selected_destinations() or []
            search_term = input.destination_search() or ""
            category_filter = input.category_filter() if hasattr(input, 'category_filter') else ""

            filtered_db = destinations_db.copy()

            if category_filter:
                filtered_db = filtered_db[filtered_db['category'] == category_filter]

            if search_term:
                mask = filtered_db['name'].str.contains(search_term, case=False, na=False)
                if 'description' in filtered_db.columns:
                    mask |= filtered_db['description'].str.contains(search_term, case=False, na=False)
                filtered_db = filtered_db[mask]

            current_view_indices = [str(idx) for idx, row in filtered_db.iterrows()]
            updated_selections = [sel for sel in persistent_selections if sel not in current_view_indices]
            updated_selections.extend(current_selections)
            updated_selections = list(dict.fromkeys(updated_selections))

            selected_destinations.set(updated_selections)

    # Clear all selections (RESTORED)
    @reactive.Effect
    @reactive.event(input.clear_selections)
    def _():
        selected_destinations.set([])
        shiny.ui.update_checkbox_group(
            session=session,
            id="selected_dest_checkboxes",
            selected=[]
        )

    # Render destination checkboxes with search functionality (RESTORED)
    @render.ui
    def destination_checkboxes():
        if not destinations_available:
            return ui.div()

        search_term = input.destination_search() or ""
        category_filter = input.category_filter() if hasattr(input, 'category_filter') else ""

        filtered_db = destinations_db.copy()

        if category_filter:
            filtered_db = filtered_db[filtered_db['category'] == category_filter]

        if search_term:
            mask = filtered_db['name'].str.contains(search_term, case=False, na=False)
            if 'description' in filtered_db.columns:
                mask |= filtered_db['description'].str.contains(search_term, case=False, na=False)
            filtered_db = filtered_db[mask]

        choices = {}
        for idx, row in filtered_db.iterrows():
            display_name = row['name']
            if 'description' in row and pd.notna(row['description']):
                display_name += f" - {row['description']}"
            choices[str(idx)] = display_name

        persistent_selections = selected_destinations() or []
        current_view_selections = [s for s in persistent_selections if s in choices.keys()]

        total_selections = len(persistent_selections)
        view_selections = len(current_view_selections)

        info_text = ""
        if total_selections > 0:
            if view_selections < total_selections:
                info_text = f"Showing {view_selections} of {total_selections} selected destinations in current view"
            else:
                info_text = f"All {total_selections} selected destinations are shown"

        return ui.div(
            ui.p(info_text, style="font-size: 12px; color: #666; margin-bottom: 5px;") if info_text else ui.div(),
            ui.input_checkbox_group(
                id="selected_dest_checkboxes",
                label="Select destinations to visit:",
                choices=choices,
                selected=current_view_selections
            ),
            style="max-height: 400px; overflow-y: auto;"
        )

    # Display selected destinations (RESTORED)
    @render.ui
    def selected_destinations_display():
        selected = selected_destinations()
        if not selected:
            return ui.p("No destinations selected", style="color: #999; font-style: italic;")

        dest_list = []
        for idx_str in selected:
            idx = int(idx_str)
            name = destinations_db.iloc[idx]['name']
            dest_list.append(f"• {name}")

        return ui.div(
            ui.HTML(f"<div style='background: #e8f5e9; padding: 10px; border-radius: 5px;'>"
                   f"<strong>Selected: {len(selected)} destination(s)</strong><br>"
                   f"{'<br>'.join(dest_list)}</div>")
        )

    # Category filter effect (RESTORED)
    @reactive.Effect
    @reactive.event(input.category_filter)
    def _():
        pass

    # Search effect (RESTORED)
    @reactive.Effect
    @reactive.event(input.destination_search)
    def _():
        pass

    # Handle file uploads for upload method
    @reactive.Effect
    @reactive.event(input.source)
    def _():
        file_info = input.source()
        if file_info:
            uploaded_files.set({**uploaded_files(), 'source': file_info[0]['datapath']})

    @reactive.Effect
    @reactive.event(input.final_stop)
    def _():
        file_info = input.final_stop()
        if file_info:
            uploaded_files.set({**uploaded_files(), 'final_stop': file_info[0]['datapath']})

    @reactive.Effect
    @reactive.event(input.destinations)
    def _():
        file_info = input.destinations()
        if file_info:
            file_path = file_info[0]['datapath']
            uploaded_files.set({**uploaded_files(), 'destinations': file_path})
            
            try:
                df = pd.read_csv(file_path)
                choices = ["Select field..."] + list(df.columns)
                dest_id_choices.set(choices)
                ui.update_select("dest_id_field", choices=dict(zip(choices, choices)))
            except Exception as e:
                ui.notification_show(f"Error reading destinations file: {str(e)}", type="error")

    @reactive.Effect
    @reactive.event(input.dest_id_field)
    def _():
        field = input.dest_id_field()
        if field and field != "Select field...":
            dest_id_field.set(field)

    # Map click mode handlers (RESTORED)
    @reactive.Effect
    @reactive.event(input.set_source_mode)
    def _():
        map_click_mode.set("source")
        ui.notification_show("Click on the map to set starting point", type="message")

    @reactive.Effect
    @reactive.event(input.set_final_mode)
    def _():
        map_click_mode.set("final")
        ui.notification_show("Click on the map to set final stop", type="message")

    @reactive.Effect
    @reactive.event(input.set_dest_mode)
    def _():
        map_click_mode.set("destination")
        ui.notification_show("Click on the map to add destinations", type="message")

    @reactive.Effect
    @reactive.event(input.clear_map_points)
    def _():
        map_clicked_points.set({'source': None, 'final_stop': None, 'destinations': []})
        map_click_mode.set("none")
        ui.notification_show("All map points cleared", type="message")

    # Handle map clicks (RESTORED)
    @reactive.Effect
    @reactive.event(input.map_clicked_coords)
    def _():
        coords = input.map_clicked_coords()
        mode = map_click_mode()
        if not coords or mode == "none":
            return

        lat, lng = coords['lat'], coords['lng']

        if mode == "source":
            ui.modal_show(ui.modal(
                ui.h4("Name the Starting Point"),
                ui.input_text("point_name_input", "Enter name:", placeholder="e.g., My Office"),
                ui.p(f"Coordinates: {lat:.6f}, {lng:.6f}", style="color: #666; font-size: 12px;"),
                footer=[ui.input_action_button("save_source_point", "Save", class_="btn-success"), ui.modal_button("Cancel")],
                easy_close=True
            ))
        elif mode == "final":
            ui.modal_show(ui.modal(
                ui.h4("Name the Final Stop"),
                ui.input_text("point_name_input", "Enter name:", placeholder="e.g., Airport"),
                ui.p(f"Coordinates: {lat:.6f}, {lng:.6f}", style="color: #666; font-size: 12px;"),
                footer=[ui.input_action_button("save_final_point", "Save", class_="btn-warning"), ui.modal_button("Cancel")],
                easy_close=True
            ))
        elif mode == "destination":
            ui.modal_show(ui.modal(
                ui.h4("Name the Destination"),
                ui.input_text("point_name_input", "Enter name:", placeholder="e.g., Tourist Site"),
                ui.p(f"Coordinates: {lat:.6f}, {lng:.6f}", style="color: #666; font-size: 12px;"),
                footer=[ui.input_action_button("save_dest_point", "Add", class_="btn-info"), ui.modal_button("Cancel")],
                easy_close=True
            ))

    # Save clicked points (RESTORED)
    @reactive.Effect
    @reactive.event(input.save_source_point)
    def _():
        coords = input.map_clicked_coords()
        name = input.point_name_input() or "Starting Point"
        if coords:
            current_points = map_clicked_points()
            current_points['source'] = {'name': name, 'latitude': coords['lat'], 'longitude': coords['lng']}
            map_clicked_points.set(current_points)
            map_click_mode.set("none")
            ui.modal_remove()
            ui.notification_show(f"Starting point '{name}' saved!", type="success")
            rerender_trigger.set(rerender_trigger() + 1)

    @reactive.Effect
    @reactive.event(input.save_final_point)
    def _():
        coords = input.map_clicked_coords()
        name = input.point_name_input() or "Final Stop"
        if coords:
            current_points = map_clicked_points()
            current_points['final_stop'] = {'name': name, 'latitude': coords['lat'], 'longitude': coords['lng']}
            map_clicked_points.set(current_points)
            map_click_mode.set("none")
            ui.modal_remove()
            ui.notification_show(f"Final stop '{name}' saved!", type="success")
            rerender_trigger.set(rerender_trigger() + 1)

    @reactive.Effect
    @reactive.event(input.save_dest_point)
    def _():
        coords = input.map_clicked_coords()
        name = input.point_name_input() or f"Destination {len(map_clicked_points()['destinations']) + 1}"
        if coords:
            current_points = map_clicked_points()
            current_points['destinations'].append({'name': name, 'latitude': coords['lat'], 'longitude': coords['lng']})
            map_clicked_points.set(current_points)
            ui.modal_remove()
            ui.notification_show(f"Destination '{name}' added!", type="success")
            rerender_trigger.set(rerender_trigger() + 1)

    # Click mode status (RESTORED)
    @render.ui
    def click_mode_status():
        mode = map_click_mode()
        if mode == "source":
            return ui.div(ui.p("🟢 Click map to set STARTING POINT", style="color: green; font-weight: bold;"))
        elif mode == "final":
            return ui.div(ui.p("🟡 Click map to set FINAL STOP", style="color: orange; font-weight: bold;"))
        elif mode == "destination":
            return ui.div(ui.p("🔵 Click map to ADD DESTINATIONS", style="color: blue; font-weight: bold;"))
        return ui.div(ui.p("Click a button above to start", style="color: #666;"))

    # Map points summary (RESTORED)
    @render.ui
    def map_points_summary():
        points = map_clicked_points()
        source_status = "✓ Set" if points['source'] else "✗ Not set"
        final_status = "✓ Set" if points['final_stop'] else "✗ Not set"
        dest_count = len(points['destinations'])
        
        return ui.div(
            ui.p(f"Starting point: {source_status}", style="margin: 2px 0;"),
            ui.p(f"Final stop: {final_status}", style="margin: 2px 0;"),
            ui.p(f"Destinations: {dest_count}", style="margin: 2px 0;"),
            style="font-size: 13px; padding: 10px; background: #f8f9fa; border-radius: 5px; margin-top: 10px;"
        )

    # Process route calculation with constraints
    @reactive.Effect
    @reactive.event(input.processButton)
    def _():
        try:
            method = input.input_method()
            
            # Prepare data based on input method
            if method == "upload":
                files = uploaded_files()
                if not all(files.values()):
                    ui.notification_show("Please upload all required files", type="error")
                    return
                
                source_df = pd.read_csv(files['source'])
                final_stop_df = pd.read_csv(files['final_stop'])
                destinations_df = pd.read_csv(files['destinations'])
                id_field = dest_id_field()
                
                if not id_field:
                    ui.notification_show("Please select a destination ID field", type="error")
                    return
                    
            elif method == "select":
                if not destinations_available:
                    ui.notification_show("Database not available", type="error")
                    return
                
                source_idx = input.source_select()
                final_idx = input.final_stop_select()
                selected = selected_destinations()
                
                if not source_idx or not final_idx or not selected:
                    ui.notification_show("Please select starting point, final stop, and at least one destination", type="error")
                    return
                
                source_row = destinations_db.iloc[int(source_idx)]
                source_df = pd.DataFrame({
                    'latitude': [source_row['latitude']],
                    'longitude': [source_row['longitude']],
                    'name': [source_row['name']]
                })

                final_row = destinations_db.iloc[int(final_idx)]
                final_stop_df = pd.DataFrame({
                    'latitude': [final_row['latitude']],
                    'longitude': [final_row['longitude']],
                    'name': [final_row['name']]
                })

                dest_rows = []
                for idx_str in selected:
                    idx = int(idx_str)
                    row = destinations_db.iloc[idx]
                    dest_rows.append({
                        'latitude': row['latitude'],
                        'longitude': row['longitude'],
                        'name': row['name']
                    })
                destinations_df = pd.DataFrame(dest_rows)
                id_field = 'name'
                
            elif method == "map_click":
                points = map_clicked_points()
                if not points['source'] or not points['final_stop'] or not points['destinations']:
                    ui.notification_show("Please set starting point, final stop, and at least one destination", type="error")
                    return
                
                source_df = pd.DataFrame({
                    'latitude': [points['source']['latitude']],
                    'longitude': [points['source']['longitude']],
                    'name': [points['source']['name']]
                })

                final_stop_df = pd.DataFrame({
                    'latitude': [points['final_stop']['latitude']],
                    'longitude': [points['final_stop']['longitude']],
                    'name': [points['final_stop']['name']]
                })

                dest_rows = []
                for dest in points['destinations']:
                    dest_rows.append({
                        'latitude': dest['latitude'],
                        'longitude': dest['longitude'],
                        'name': dest['name']
                    })
                destinations_df = pd.DataFrame(dest_rows)
                id_field = 'name'
            
            # Prepare constraints
            bridges_data = None
            if input.use_bridge_constraint() and bridges_available:
                bridges_data = bridges
            
            priority_rule = None
            if input.use_priority_rule():
                try:
                    before_idx_str = input.priority_before()
                    after_idx_str = input.priority_after()
                    
                    if before_idx_str and after_idx_str and before_idx_str != "" and after_idx_str != "":
                        before_idx = int(before_idx_str)
                        after_idx = int(after_idx_str)
                        
                        if before_idx == after_idx:
                            ui.notification_show("Please select different destinations for priority rule", type="warning")
                            return
                        
                        before_name = destinations_df.iloc[before_idx][id_field]
                        after_name = destinations_df.iloc[after_idx][id_field]
                        priority_rule = {
                            'before_dest': before_name,
                            'after_dest': after_name,
                            'before_idx': before_idx,
                            'after_idx': after_idx
                        }
                except Exception as e:
                    print(f"Priority rule error: {e}")
                    pass
            
            # Calculate route with constraints
            ui.notification_show("Calculating optimal route...", type="message", duration=3)

            result.set(None)
            calculate_route_task.invoke(
                source_df, 
                destinations_df, 
                id_field, 
                final_stop_df,
                bridges_data,
                priority_rule
            )
            
        except ValueError as e:
            ui.notification_show(str(e), type="error", duration=10)
        except Exception as e:
            ui.notification_show(f"Error calculating route: {str(e)}", type="error", duration=10)
            import traceback
            print(traceback.format_exc())

    # Google Maps navigation button (NEW)
    @render.ui
    def google_maps_button():
        result_value = result()
        if result_value is None:
            return None
        
        # Build Google Maps URL with waypoints
        source = result_value['source']
        final_stop = result_value['final_stop']
        destinations = result_value['destinations']
        
        origin = f"{source.geometry.y.iloc[0]},{source.geometry.x.iloc[0]}"
        destination = f"{final_stop.geometry.y.iloc[0]},{final_stop.geometry.x.iloc[0]}"
        
        # Get waypoints in order
        waypoints = []
        for _, row in destinations.iterrows():
            waypoints.append(f"{row.geometry.y},{row.geometry.x}")
        
        waypoints_str = "|".join(waypoints)
        
        # Construct Google Maps URL
        base_url = "https://www.google.com/maps/dir/"
        params = f"?api=1&origin={origin}&destination={destination}&waypoints={waypoints_str}&travelmode=driving"
        gmaps_url = base_url + params
        
        return ui.a(
            "🗺️ Open in Google Maps",
            href=gmaps_url,
            target="_blank",
            class_="btn btn-sm btn-primary"
        )

    # Enhanced map with colored routes (RESTORED + NEW)
    @render.ui
    def map():
        _ = rerender_trigger()  # Force rerender when new points are clicked
        try:
            m = folium.Map(location=center_coords, zoom_start=8.5)
            
            # Remove default OSM layer
            for key in list(m._children.keys()):
                if key.startswith('openstreetmap'):
                    del m._children[key]
            
            # Add basemaps
            folium.TileLayer('CartoDB positron', name='CartoDB Positron').add_to(m)
            folium.TileLayer(
                tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
                attr='Google',
                name='Google Satellite'
            ).add_to(m)
            folium.TileLayer(
                tiles='https://mt1.google.com/vt/lyrs=p&x={x}&y={y}&z={z}',
                attr='Google',
                name='Google Terrain'
            ).add_to(m)
            folium.TileLayer('OpenStreetMap', name='Open Street Map').add_to(m)
            
            # Style functions
            def style_country(feature):
                return {'fillColor': '#acbbb4', 'color': '#3f4b46', 'weight': 4, 'fillOpacity': 0.2}
            
            def style_lakes(feature):
                return {'fillColor': '#37a3bd', 'color': '#345a6a', 'weight': 1, 'fillOpacity': 0.6}
            
            def style_np(feature):
                return {'fillColor': '#13764b', 'color': '#006600', 'weight': 2, 'fillOpacity': 0.6}
            
            # Add layers
            folium.GeoJson(country, style_function=style_country, name="Country Border", control=False).add_to(m)
            folium.GeoJson(np_parks, style_function=style_np, name="National Parks", control=False).add_to(m)
            folium.GeoJson(lakes, style_function=style_lakes, name="Lakes", control=False).add_to(m)
            
            # Add bridges if available
            if bridges_available:
                bridges_copy = bridges.copy()
                bridges_copy['status'] = bridges_copy['status'].astype(str)
                operational = bridges_copy[bridges_copy['status'].str.lower() == 'operational']
                non_operational = bridges_copy[bridges_copy['status'].str.lower() == 'non-operational']
                
                for _, bridge in operational.iterrows():
                    folium.CircleMarker(
                        location=[bridge.geometry.y, bridge.geometry.x],
                        radius=4,
                        color='green',
                        fill=True,
                        fillColor='green',
                        fillOpacity=0.7,
                        popup="Operational Bridge",
                        tooltip="Bridge: Operational"
                    ).add_to(m)
                
                for _, bridge in non_operational.iterrows():
                    folium.CircleMarker(
                        location=[bridge.geometry.y, bridge.geometry.x],
                        radius=4,
                        color='red',
                        fill=True,
                        fillColor='red',
                        fillOpacity=0.7,
                        popup="Non-Operational Bridge",
                        tooltip="Bridge: NON-OPERATIONAL"
                    ).add_to(m)
            
            # Create dynamic markers group
            dynamic_markers = folium.FeatureGroup(name="Dynamic Markers", control=False).add_to(m)
            
            # Add map click functionality (RESTORED)
            if input.input_method() == "map_click":
                map_name = m.get_name()
                click_macro = Template(f"""
                    {{% macro script(this, kwargs) %}}
                    function getLatLng(e) {{
                        var lat = e.latlng.lat.toFixed(6),
                            lng = e.latlng.lng.toFixed(6);
                        parent.Shiny.setInputValue('map_clicked_coords', {{
                            lat: parseFloat(lat),
                            lng: parseFloat(lng),
                            timestamp: new Date().getTime()
                        }}, {{priority: 'event'}});
                    }}
                    {map_name}.on('click', getLatLng);
                    {{% endmacro %}}
                """)

                el = MacroElement()
                el._template = click_macro
                m.get_root().add_child(el)

                # Add clicked points to map (only if no optimal route is calculated yet)
                if result() is None:
                    points = map_clicked_points()
                    if points['source']:
                        folium.Marker(
                            location=[points['source']['latitude'], points['source']['longitude']],
                            popup=f"Starting Point: {points['source']['name']}",
                            icon=folium.Icon(color='green', icon='play')
                        ).add_to(dynamic_markers)
                    
                    if points['final_stop']:
                        folium.Marker(
                            location=[points['final_stop']['latitude'], points['final_stop']['longitude']],
                            popup=f"Final Stop: {points['final_stop']['name']}",
                            icon=folium.Icon(color='red', icon='stop')
                        ).add_to(dynamic_markers)
                    
                    for i, dest in enumerate(points['destinations']):
                        folium.Marker(
                            location=[dest['latitude'], dest['longitude']],
                            popup=f"Destination {i+1}: {dest['name']}",
                            icon=folium.Icon(color='blue', icon='info-sign')
                        ).add_to(dynamic_markers)
            
            # Add database selections for select method (RESTORED)
            elif input.input_method() == "select" and destinations_available and result() is None:
                source_idx = get_optional_input("source_select", "")
                final_idx = get_optional_input("final_stop_select", "")
                selected = selected_destinations()
                
                if source_idx:
                    source_row = destinations_db.loc[int(source_idx)]
                    folium.Marker(
                        location=[source_row['latitude'], source_row['longitude']],
                        popup=f"Starting Point: {source_row['name']}",
                        tooltip=source_row['name'],
                        icon=folium.Icon(color='green', icon='play')
                    ).add_to(dynamic_markers)
                
                if final_idx:
                    final_row = destinations_db.loc[int(final_idx)]
                    folium.Marker(
                        location=[final_row['latitude'], final_row['longitude']],
                        popup=f"Final Stop: {final_row['name']}",
                        tooltip=final_row['name'],
                        icon=folium.Icon(color='red', icon='stop')
                    ).add_to(dynamic_markers)
                
                for i, idx in enumerate(selected):
                    dest = destinations_db.loc[int(idx)]
                    folium.Marker(
                        location=[dest['latitude'], dest['longitude']],
                        popup=f"Destination {i+1}: {dest['name']}",
                        tooltip=dest['name'],
                        icon=folium.Icon(color='blue', icon='info-sign')
                    ).add_to(dynamic_markers)
            
            # Draw calculated route with color coding (NEW)
            result_value = result()
            if result_value is not None:
                route_detailed = result_value['route_detailed']
                
                # Group segments by pavement status and draw with different colors
                for pavement_type in ['Paved', 'Unpaved', 'Unknown']:
                    subset = route_detailed[route_detailed['surface'] == pavement_type]
                    
                    if len(subset) > 0:
                        if pavement_type == 'Paved':
                            color = '#2ECC71'  # Green
                            weight = 5
                        elif pavement_type == 'Unpaved':
                            color = '#E67E22'  # Orange
                            weight = 5
                        else:  # Unknown
                            color = '#800000'  # Maroon
                            weight = 4
                        
                        for _, row in subset.iterrows():
                            if row.geometry.geom_type == 'LineString':
                                coords = [(coord[1], coord[0]) for coord in row.geometry.coords]
                                folium.PolyLine(
                                    coords,
                                    color=color,
                                    weight=weight,
                                    opacity=0.8,
                                    popup=f"Surface: {pavement_type}"
                                ).add_to(m)
                            elif row.geometry.geom_type == 'MultiLineString':
                                for line in row.geometry.geoms:
                                    coords = [(coord[1], coord[0]) for coord in line.coords]
                                    folium.PolyLine(
                                        coords,
                                        color=color,
                                        weight=weight,
                                        opacity=0.8,
                                        popup=f"Surface: {pavement_type}"
                                    ).add_to(m)
                
                # Add markers for route points
                folium.Marker(
                    location=[float(result_value['source'].geometry.y.iloc[0]), 
                             float(result_value['source'].geometry.x.iloc[0])],
                    popup='Starting point',
                    icon=folium.Icon(color='lightgreen', icon='play')
                ).add_to(m)
                
                folium.Marker(
                    location=[float(result_value['final_stop'].geometry.y.iloc[0]), 
                             float(result_value['final_stop'].geometry.x.iloc[0])],
                    popup='Final Stop',
                    icon=folium.Icon(color='darkpurple', icon='stop')
                ).add_to(m)
                
                dest_id = result_value['used_id_field']
                for _, row in result_value['destinations'].iterrows():
                    destination_name = str(row[dest_id]) if dest_id in row else "Destination"
                    route_order = row.get('route_order', 'Destination')
                    popup_text = f"{route_order}: {destination_name}"
                    tooltip_text = popup_text
                    
                    folium.Marker(
                        location=[float(row.geometry.y), float(row.geometry.x)],
                        popup=popup_text,
                        tooltip=tooltip_text,
                        icon=folium.Icon(color='blue', icon='flag')
                    ).add_to(m)
            
            folium.LayerControl(collapsed=True, overlays=True).add_to(m)
            
            return ui.HTML(m._repr_html_())
        
        except Exception as e:
            print(f"Error in map function: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return ui.HTML(f"<p>An error occurred while creating the map: {str(e)}</p>")

    # Download handlers (RESTORED)
    @render.download(filename="my_route.gpkg")
    def downloadRoute():
        result_value = result()
        if result_value is None:
            return
        
        route = result_value['route_detailed']
        tmp_dir = tempfile.gettempdir()
        tmp_filename = f"route_{os.getpid()}_{int(time.time())}.gpkg"
        tmp_path = os.path.join(tmp_dir, tmp_filename)
        
        try:
            route.to_file(tmp_path, layer="optimal_route", driver="GPKG")
            with open(tmp_path, 'rb') as f:
                content = f.read()
            yield content
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    @render.download(filename="route_segments.csv")
    def downloadRouteSegments():
        result_value = result()
        if result_value is None:
            return
        route_segments = result_value['route_segments']
        yield route_segments.to_csv(index=False)

    # Show route segments modal (RESTORED)
    @reactive.Effect
    @reactive.event(input.Show_Segments_Table)
    def _():
        result_value = result()
        if result_value is None:
            ui.notification_show("Please calculate a route first", type="warning")
            return
        
        route_segments = result_value['route_segments']
        
        table_html = "<div style='max-height: 400px; overflow-y: auto;'>"
        table_html += "<table class='table table-striped table-sm'>"
        table_html += "<thead><tr>"
        
        for col in route_segments.columns:
            table_html += f"<th>{col}</th>"
        table_html += "</tr></thead><tbody>"
        
        for _, row in route_segments.iterrows():
            table_html += "<tr>"
            for col in route_segments.columns:
                value = row[col]
                if isinstance(value, float):
                    value = f"{value:.2f}"
                table_html += f"<td>{value}</td>"
            table_html += "</tr>"
        
        table_html += "</tbody></table></div>"
        
        ui.modal_show(
            ui.modal(
                ui.h3("Route Segments"),
                ui.HTML(table_html),
                size="xl",
                easy_close=True,
                footer=ui.modal_button("Close")
            )
        )

    # Show road surface statistics modal with color coding (RESTORED + NEW)
    @reactive.Effect
    @reactive.event(input.Show_Table)
    def _():
        result_value = result()
        if result_value is None:
            ui.notification_show("Please calculate a route first", type="warning")
            return
        
        route_detailed = result_value['route_detailed']
        df = pd.DataFrame(route_detailed.drop(columns='geometry'))
        
        surface_stats = df.groupby('surface')['segment_length'].sum().reset_index()
        surface_stats = surface_stats.rename(columns={'segment_length': 'total_length_m'})
        surface_stats['total_length_km'] = surface_stats['total_length_m'] / 1000
        total_length = surface_stats['total_length_km'].sum()
        surface_stats['percentage'] = (surface_stats['total_length_km'] / total_length) * 100
        surface_stats['total_length_km'] = surface_stats['total_length_km'].round(2)
        surface_stats['percentage'] = surface_stats['percentage'].round(2)
        surface_stats = surface_stats.sort_values('total_length_km', ascending=False)
        
        table_html = "<table class='table table-striped'>"
        table_html += "<thead><tr><th>Surface Type</th><th>Length (km)</th><th>Percentage</th></tr></thead><tbody>"
        
        for _, row in surface_stats.iterrows():
            # Add color coding to match route colors
            if row['surface'] == 'Paved':
                color_style = "background-color: #D5F4E6;"
            elif row['surface'] == 'Unpaved':
                color_style = "background-color: #FADBD8;"
            else:
                color_style = "background-color: #F9E79F;"
            
            table_html += f"<tr style='{color_style}'><td><strong>{row['surface']}</strong></td><td>{row['total_length_km']}</td><td>{row['percentage']}%</td></tr>"
        
        table_html += f"</tbody></table>"
        table_html += f"<p><strong>Total Route Length: {total_length:.2f} km</strong></p>"
        
        ui.modal_show(
            ui.modal(
                ui.h3("Road Surface Statistics"),
                ui.HTML(table_html),
                size="l",
                easy_close=True,
                footer=ui.modal_button("Close")
            )
        )

app = App(app_ui, server)
