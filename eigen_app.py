import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import numpy as np
from scipy.spatial import Delaunay
import yaml

# Import documentation content
from eigen_documentation import DOCS_CONTENT
from graph_utils import (
    empty_graph,
    default_graph,
    render_tab_content,
    update_graph_store
)
from callbacks import register_callbacks


# -------------------------
# Dash App Setup
# -------------------------
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Build all controls as Dash components (factory)
def build_control(control_id, control_cfg):
    t = control_cfg.get("type")
    label = control_cfg.get("label", control_id)
    if t == "button":
        return html.Button(label, id=control_cfg["id"], n_clicks=0)
    elif t == "slider":
        return dcc.Slider(
            id=control_cfg["id"],
            min=control_cfg.get("min", 0),
            max=control_cfg.get("max", 1),
            step=control_cfg.get("step", 0.1),
            value=control_cfg.get("value", 0.5),
            marks={float(control_cfg.get("min", 0)): str(control_cfg.get("min", 0)), float(control_cfg.get("max", 1)): str(control_cfg.get("max", 1))}
        )
    elif t == "input":
        return dcc.Input(
            id=control_cfg["id"],
            type="number" if isinstance(control_cfg.get("value", 0), (int, float)) else "text",
            value=control_cfg.get("value", ""),
            step=control_cfg.get("step"),
            min=control_cfg.get("min"),
        )
    elif t == "dropdown":
        return dcc.Dropdown(
            id=control_cfg["id"],
            options=[{"label": str(opt), "value": opt} for opt in control_cfg.get("options", [])],
            value=control_cfg.get("value")
        )
    elif t == "textarea":
        return dcc.Textarea(
            id=control_cfg["id"],
            value=control_cfg.get("value", ""),
            style={"width": "100%", "height": f"{control_cfg.get('height', 100)}px"}
        )
    else:
        return html.Div(f"Unknown control type: {t}", id=control_cfg["id"])

all_controls = {cid: build_control(cid, cfg) for cid, cfg in config["controls"].items()}

# Layout: all controls always present, hidden unless active tab
controls_layout = html.Div([
    html.Div(
        all_controls[control_id],
        id=f"wrapper-{control_id}",
        style={"display": "none"}
    )
    for control_id in all_controls
])

tabs = [
    dcc.Tab(label=tab["name"], value=tab["value"]) for tab in config["tabs"]
]

app.layout = html.Div([
    dcc.Tabs(id="tabs", value=config["tabs"][0]["value"], children=tabs),
    controls_layout,
    html.Div(id='tab-content'),
    html.Hr(),
    html.Div([
        html.H5("Debug: graph-store"),
        html.Pre(id='graph-store-debug', style={"fontSize": "10px", "maxHeight": "200px", "overflowY": "auto", "background": "#f8f8f8"}),
        html.H5("Debug: clustering-store"),
        html.Pre(id='clustering-store-debug', style={"fontSize": "10px", "maxHeight": "200px", "overflowY": "auto", "background": "#f8f8f8"})
    ]),
    *[dcc.Store(id=store["id"]) for store in config["stores"]]
], style={"display": "flex", "height": "100vh"})

register_callbacks(app)

# Callback to show/hide controls based on tab
@app.callback(
    [Output(f"wrapper-{cid}", "style") for cid in all_controls],
    Input("tabs", "value")
)
def toggle_controls(tab_value):
    tab_config = next(tab for tab in config["tabs"] if tab["value"] == tab_value)
    visible = set(tab_config["controls"])
    return [
        {"display": "block"} if cid in visible else {"display": "none"}
        for cid in all_controls
    ]

if __name__ == '__main__':
    app.run(debug=True)