import dash
from dash import dcc, html

from graph_utils import default_graph
from callbacks import register_callbacks

app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

app.layout = html.Div([
    dcc.Tabs(
        id="tabs",
        value="tab-graph",
        children=[
            dcc.Tab(label="Graph Tools", value="tab-graph"),
            dcc.Tab(label="Clustering", value="tab-clustering"),
            dcc.Tab(label="Documentation", value="tab-docs"),
        ],
    ),
    html.Div(id="tab-content"),
    html.Hr(),
    html.Div(
        [
            html.H5("Debug: graph-tools-store"),
            html.Pre(
                id="graph-store-debug",
                style={
                    "fontSize": "10px",
                    "maxHeight": "200px",
                    "overflowY": "auto",
                    "background": "#f8f8f8",
                },
            ),
            html.H5("Debug: clustering-store"),
            html.Pre(
                id="clustering-store-debug",
                style={
                    "fontSize": "10px",
                    "maxHeight": "200px",
                    "overflowY": "auto",
                    "background": "#f8f8f8",
                },
            ),
        ]
    ),
    dcc.Store(id="graph-tools-store", data=default_graph()),
    dcc.Store(id="clustering-store"),
])

register_callbacks(app)

if __name__ == "__main__":
    app.run(debug=True)
