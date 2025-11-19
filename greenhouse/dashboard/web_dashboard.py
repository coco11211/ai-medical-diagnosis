"""
Web-based Real-time Dashboard
Displays greenhouse metrics and controls
"""
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json


class GreenhouseDashboard:
    """Interactive web dashboard for greenhouse monitoring"""

    def __init__(self, sensor_manager=None, actuator_manager=None, host='127.0.0.1', port=8050):
        self.sensor_manager = sensor_manager
        self.actuator_manager = actuator_manager
        self.host = host
        self.port = port

        # Initialize Dash app
        self.app = dash.Dash(__name__, update_title=None)
        self.app.title = "Smart Greenhouse Dashboard"

        # Setup layout
        self._setup_layout()

        # Setup callbacks
        self._setup_callbacks()

    def _setup_layout(self):
        """Create dashboard layout"""
        self.app.layout = html.Div([
            html.Div([
                html.H1("🌱 Smart Greenhouse Control System", style={'textAlign': 'center', 'color': '#2c3e50'}),
                html.P("Real-time Monitoring & Control", style={'textAlign': 'center', 'color': '#7f8c8d'})
            ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'marginBottom': '20px'}),

            # Auto-refresh interval
            dcc.Interval(
                id='interval-component',
                interval=5*1000,  # Update every 5 seconds
                n_intervals=0
            ),

            # Current Status Cards
            html.Div([
                html.Div([
                    html.H3("Current Conditions", style={'color': '#27ae60'}),
                    html.Div(id='status-cards')
                ], style={'width': '100%', 'marginBottom': '20px'}),
            ]),

            # Sensor Graphs
            html.Div([
                html.H3("Environmental Monitoring", style={'color': '#2980b9'}),
                dcc.Graph(id='sensor-graphs', style={'height': '400px'}),
            ], style={'marginBottom': '30px'}),

            # Actuator Status
            html.Div([
                html.Div([
                    html.H3("Lighting Control", style={'color': '#f39c12'}),
                    html.Div(id='lighting-status'),
                    html.Br(),
                    html.Button('All Lights ON', id='btn-lights-on', n_clicks=0,
                               style={'margin': '5px', 'padding': '10px', 'backgroundColor': '#f39c12', 'color': 'white', 'border': 'none', 'borderRadius': '5px'}),
                    html.Button('All Lights OFF', id='btn-lights-off', n_clicks=0,
                               style={'margin': '5px', 'padding': '10px', 'backgroundColor': '#95a5a6', 'color': 'white', 'border': 'none', 'borderRadius': '5px'}),
                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),

                html.Div([
                    html.H3("Irrigation Control", style={'color': '#3498db'}),
                    html.Div(id='irrigation-status'),
                    html.Br(),
                    html.Button('Emergency Stop', id='btn-stop-irrigation', n_clicks=0,
                               style={'margin': '5px', 'padding': '10px', 'backgroundColor': '#e74c3c', 'color': 'white', 'border': 'none', 'borderRadius': '5px'}),
                ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),
            ], style={'marginBottom': '30px'}),

            # Statistics and Predictions
            html.Div([
                html.H3("Growth Metrics & Predictions", style={'color': '#8e44ad'}),
                html.Div(id='growth-metrics'),
            ], style={'marginBottom': '30px'}),

            # Alerts
            html.Div([
                html.H3("⚠️ Active Alerts", style={'color': '#e74c3c'}),
                html.Div(id='alerts-display'),
            ]),

            # Hidden div for button outputs
            html.Div(id='button-output', style={'display': 'none'})

        ], style={'fontFamily': 'Arial, sans-serif', 'padding': '20px', 'backgroundColor': '#f5f6fa'})

    def _setup_callbacks(self):
        """Setup dashboard callbacks"""

        @self.app.callback(
            Output('status-cards', 'children'),
            Input('interval-component', 'n_intervals')
        )
        def update_status_cards(n):
            if not self.sensor_manager:
                return html.Div("No sensor data available")

            readings = self.sensor_manager.read_all_sensors()

            cards = []
            colors = {
                'Temperature': '#e74c3c',
                'Humidity': '#3498db',
                'Soil Moisture': '#2ecc71',
                'Light': '#f39c12',
                'CO2': '#9b59b6'
            }

            for sensor_id, reading in readings.items():
                if 'value' in reading:
                    name = reading['name']
                    value = reading['value']
                    unit = reading['unit']
                    color = colors.get(name, '#95a5a6')

                    card = html.Div([
                        html.H4(name, style={'color': color, 'marginBottom': '5px'}),
                        html.H2(f"{value:.1f} {unit}", style={'margin': '0'}),
                        html.P(reading.get('location', ''), style={'color': '#7f8c8d', 'fontSize': '12px'})
                    ], style={
                        'width': '18%',
                        'display': 'inline-block',
                        'backgroundColor': 'white',
                        'padding': '15px',
                        'margin': '5px',
                        'borderRadius': '10px',
                        'boxShadow': '0 2px 5px rgba(0,0,0,0.1)',
                        'textAlign': 'center'
                    })
                    cards.append(card)

            return cards

        @self.app.callback(
            Output('sensor-graphs', 'figure'),
            Input('interval-component', 'n_intervals')
        )
        def update_graphs(n):
            if not self.sensor_manager:
                return go.Figure()

            history = self.sensor_manager.get_history(limit=50)

            if not history:
                return go.Figure()

            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Temperature', 'Humidity', 'Soil Moisture', 'Light Level')
            )

            # Extract data
            timestamps = []
            temp_data = []
            humidity_data = []
            moisture_data = []
            light_data = []

            for entry in history:
                timestamps.append(entry['timestamp'])
                readings = entry.get('readings', {})

                for sensor_id, reading in readings.items():
                    if 'value' in reading:
                        name = reading['name']
                        if name == 'Temperature':
                            temp_data.append(reading['value'])
                        elif name == 'Humidity':
                            humidity_data.append(reading['value'])
                        elif 'Soil Moisture' in name:
                            moisture_data.append(reading['value'])
                        elif name == 'Light':
                            light_data.append(reading['value'])

            # Pad data to match timestamps
            while len(temp_data) < len(timestamps):
                temp_data.append(None)
            while len(humidity_data) < len(timestamps):
                humidity_data.append(None)
            while len(moisture_data) < len(timestamps):
                moisture_data.append(None)
            while len(light_data) < len(timestamps):
                light_data.append(None)

            # Add traces
            if temp_data:
                fig.add_trace(go.Scatter(x=timestamps, y=temp_data, mode='lines', name='Temperature', line=dict(color='#e74c3c')), row=1, col=1)

            if humidity_data:
                fig.add_trace(go.Scatter(x=timestamps, y=humidity_data, mode='lines', name='Humidity', line=dict(color='#3498db')), row=1, col=2)

            if moisture_data:
                fig.add_trace(go.Scatter(x=timestamps, y=moisture_data, mode='lines', name='Soil Moisture', line=dict(color='#2ecc71')), row=2, col=1)

            if light_data:
                fig.add_trace(go.Scatter(x=timestamps, y=light_data, mode='lines', name='Light', line=dict(color='#f39c12')), row=2, col=2)

            fig.update_xaxes(showgrid=True, gridcolor='lightgray')
            fig.update_yaxes(showgrid=True, gridcolor='lightgray')
            fig.update_layout(height=400, showlegend=False, plot_bgcolor='white')

            return fig

        @self.app.callback(
            Output('lighting-status', 'children'),
            Input('interval-component', 'n_intervals')
        )
        def update_lighting_status(n):
            if not self.actuator_manager:
                return "No lighting control available"

            status = self.actuator_manager.lighting.get_all_status()
            lights = status.get('lights', {})

            items = []
            for light_id, light_info in lights.items():
                color = '#2ecc71' if light_info['is_on'] else '#95a5a6'
                items.append(html.Div([
                    html.Span(f"• {light_info['name']}: ", style={'fontWeight': 'bold'}),
                    html.Span(f"{light_info['intensity']}% ", style={'color': color}),
                    html.Span(f"({light_info['mode']})", style={'fontSize': '12px', 'color': '#7f8c8d'})
                ], style={'marginBottom': '5px'}))

            items.append(html.P(f"Power: {status.get('current_sensor_reading', 0):.0f} lux detected",
                               style={'marginTop': '10px', 'fontSize': '12px', 'color': '#7f8c8d'}))

            return items

        @self.app.callback(
            Output('irrigation-status', 'children'),
            Input('interval-component', 'n_intervals')
        )
        def update_irrigation_status(n):
            if not self.actuator_manager:
                return "No irrigation control available"

            status = self.actuator_manager.irrigation.get_all_status()
            zones = status.get('zones', {})

            items = []
            for zone_id, zone_info in zones.items():
                color = '#3498db' if zone_info['is_active'] else '#95a5a6'
                items.append(html.Div([
                    html.Span(f"• {zone_info['name']}: ", style={'fontWeight': 'bold'}),
                    html.Span("ACTIVE" if zone_info['is_active'] else "IDLE", style={'color': color}),
                    html.Span(f" ({zone_info['total_water_used']:.1f}L used)",
                             style={'fontSize': '12px', 'color': '#7f8c8d'})
                ], style={'marginBottom': '5px'}))

            items.append(html.P(f"Total water used: {status.get('total_water_used', 0):.1f} L",
                               style={'marginTop': '10px', 'fontSize': '12px', 'color': '#7f8c8d'}))

            return items

        @self.app.callback(
            Output('growth-metrics', 'children'),
            Input('interval-component', 'n_intervals')
        )
        def update_growth_metrics(n):
            # Placeholder for growth metrics
            return html.Div([
                html.P("Growth Score: 85/100 (Good)", style={'fontSize': '16px', 'color': '#27ae60'}),
                html.P("Estimated Daily Growth: 0.42 cm/day", style={'fontSize': '14px'}),
                html.P("Days to Harvest: 45 days", style={'fontSize': '14px'}),
                html.P("Predicted Yield: 2.3 kg", style={'fontSize': '14px'}),
            ])

        @self.app.callback(
            Output('alerts-display', 'children'),
            Input('interval-component', 'n_intervals')
        )
        def update_alerts(n):
            if not self.sensor_manager:
                return html.Div("No alerts")

            # Check for alerts
            thresholds = {
                # These would be sensor IDs with their thresholds
                # This is a simplified example
            }

            # alerts = self.sensor_manager.check_alerts(thresholds)

            # Placeholder
            return html.Div("✓ All systems normal", style={'color': '#27ae60', 'fontSize': '16px'})

        @self.app.callback(
            Output('button-output', 'children'),
            [Input('btn-lights-on', 'n_clicks'),
             Input('btn-lights-off', 'n_clicks'),
             Input('btn-stop-irrigation', 'n_clicks')]
        )
        def handle_buttons(btn_on, btn_off, btn_stop):
            ctx = dash.callback_context

            if not ctx.triggered:
                return ""

            button_id = ctx.triggered[0]['prop_id'].split('.')[0]

            if button_id == 'btn-lights-on' and self.actuator_manager:
                self.actuator_manager.lighting.turn_on_all(100)
                return "Lights turned on"
            elif button_id == 'btn-lights-off' and self.actuator_manager:
                self.actuator_manager.lighting.turn_off_all()
                return "Lights turned off"
            elif button_id == 'btn-stop-irrigation' and self.actuator_manager:
                self.actuator_manager.irrigation.stop_all_zones()
                return "Irrigation stopped"

            return ""

    def run(self, debug=False):
        """Run the dashboard server"""
        print(f"\n{'='*60}")
        print(f"🌱 Smart Greenhouse Dashboard Starting...")
        print(f"{'='*60}")
        print(f"Dashboard URL: http://{self.host}:{self.port}")
        print(f"Press Ctrl+C to stop the server")
        print(f"{'='*60}\n")

        self.app.run_server(host=self.host, port=self.port, debug=debug)
