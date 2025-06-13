from dash import dcc, html
import plotly.graph_objs as go
import plotly.express as px
import networkx as nx
import numpy as np
from scipy.spatial import Delaunay
import traceback


class CustomClusteringError(Exception):
    """Exception raised when user supplied clustering code fails."""


def empty_graph():
    """Return an empty graph data structure."""
    return {"nodes": {}, "edges": []}

def default_graph():
    """Generate a default scale-free graph with random node positions."""
    G = nx.scale_free_graph(10)
    nodes = {str(n): {'x': float(np.random.uniform(-5,5)),
                      'y': float(np.random.uniform(-5,5)),
                      'z': float(np.random.uniform(-5,5)),
                      'extra': float(np.random.uniform(0,10))}
             for n in G.nodes()}
    edges = [(str(u), str(v)) for u, v in G.edges()]
    return {'nodes': nodes, 'edges': edges}

def get_tab_layout(tab, graph_data, clustering_method='spectral_lpa', clustering_step=0, clustering_node_order=None, custom_code=None):
    """
    Return the layout for each tab, allowing custom order of visualization and settings modules.
    """
    # Ensure graph_data is always a dict, never None
    if graph_data is None or not graph_data.get('nodes'):
        from graph_utils import default_graph
        graph_data = default_graph()
    if tab == 'tab-graph':
        # Only return the visualization area. All controls are created once in
        # ``eigen_app.py`` and shown/hidden via wrappers, so duplicating them
        # here leads to Dash ID conflicts.
        return html.Div([
            graph_visualization_layout(graph_data)
        ])
    elif tab == 'tab-clustering':
        cluster_labels = None
        y_prime = None
        if graph_data and graph_data.get('nodes'):
            from graph_utils import run_clustering, CustomClusteringError
            alpha = 0.5
            e = -0.5
            k = 5
            try:
                cluster_labels, y_prime = run_clustering(graph_data, clustering_method, alpha=alpha, e=e, k=k, custom_code=custom_code, return_y=True)
            except CustomClusteringError as exc:
                print(exc)
                cluster_labels, y_prime = None, None

        # Only show the main graph. The associated controls are provided by the
        # global wrappers defined in ``eigen_app.py``.
        
        return html.Div([
            dcc.Graph(id="graph", style={"height": "60vh"})
        ])
    elif tab == 'tab-docs':
        from eigen_documentation import DOCS_CONTENT
        return html.Div(DOCS_CONTENT, style={"padding": "20px"})

# For backward compatibility
render_tab_content = get_tab_layout


def update_graph_store(add_node_clicks, remove_node_clicks, add_edge_clicks, remove_edge_clicks,
                       gen_clicks, clear_clicks,
                       graph_data, node_id,
                       remove_node_id, src, tgt,
                       rem_src, rem_tgt,
                       rg_family, rg_num_nodes, rg_density):
    from dash import callback_context
    ctx = callback_context
    if not ctx.triggered:
        return graph_data
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]
    if trigger == 'clear-graph':
        return empty_graph()
    if trigger == 'generate-graph':
        new_graph = {'nodes': {}, 'edges': []}
        p = rg_density if rg_density is not None else 0.3
        if rg_family == 'ER':
            G = nx.erdos_renyi_graph(rg_num_nodes, p)
        elif rg_family == 'BA':
            m = max(1, int(p * (rg_num_nodes-1))) if rg_num_nodes > 1 else 1
            G = nx.barabasi_albert_graph(rg_num_nodes, m)
        elif rg_family == 'WS':
            k_ws = max(2, int(p * (rg_num_nodes-1)))
            G = nx.watts_strogatz_graph(rg_num_nodes, k_ws, p)
        elif rg_family == 'SF':
            alpha = p
            beta = 1 - p - 0.01 if (1 - p) > 0.01 else 0.01
            gamma = 1 - alpha - beta
            if gamma <= 0:
                gamma = 0.01
                beta = 1 - alpha - gamma
            G = nx.Graph(nx.scale_free_graph(rg_num_nodes, alpha=alpha, beta=beta, gamma=gamma))
        elif rg_family == 'Star':
            G = nx.star_graph(rg_num_nodes-1)
        elif rg_family == 'Lattice':
            G = nx.grid_2d_graph(int(np.sqrt(rg_num_nodes)), int(np.sqrt(rg_num_nodes)))
            G = nx.relabel_nodes(G, {n: f"{n[0]}-{n[1]}" for n in G.nodes()})
        elif rg_family == 'Delaunay':
            points = np.random.uniform(-5,5,(rg_num_nodes,2))
            tri = Delaunay(points)
            G = nx.Graph()
            G.add_nodes_from([str(i) for i in range(rg_num_nodes)])
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i+1,3):
                        G.add_edge(str(simplex[i]), str(simplex[j]))
        for n in G.nodes():
            new_graph['nodes'][str(n)] = {'x': float(np.random.uniform(-5,5)),
                                           'y': float(np.random.uniform(-5,5)),
                                           'z': float(np.random.uniform(-5,5)),
                                           'extra': float(np.random.uniform(0,10))}
        new_graph['edges'] = [(str(u),str(v)) for u,v in G.edges()]
        return new_graph
    # Node/Edge edits
    graph_data = graph_data or {'nodes': {}, 'edges': []}
    if trigger == 'add-node':
        graph_data['nodes'][node_id] = {
            'x': 0.0, 'y': 0.0, 'z': 0.0, 'extra': 0.0
        }
    elif trigger == 'remove-node':
        if remove_node_id in graph_data['nodes']:
            del graph_data['nodes'][remove_node_id]
            graph_data['edges'] = [e for e in graph_data['edges'] if remove_node_id not in e]
    elif trigger == 'add-edge':
        if src in graph_data['nodes'] and tgt in graph_data['nodes']:
            edge = (src, tgt)
            if edge not in graph_data['edges'] and (tgt, src) not in graph_data['edges']:
                graph_data['edges'].append(edge)
    elif trigger == 'remove-edge':
        graph_data['edges'] = [e for e in graph_data['edges'] if not (e[0]==rem_src and e[1]==rem_tgt)]
    return graph_data


def run_clustering(graph_data, method, alpha=0.5, e=-0.5, k=5, custom_code=None, return_y=False):
    import networkx as nx
    import numpy as np
    nodes = list(graph_data.get('nodes', {}).keys())
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(graph_data.get('edges', []))
    node_idx = {node: i for i, node in enumerate(nodes)}
    Y = np.eye(len(nodes))
    if method == 'spectral_lpa':
        A = nx.to_numpy_array(G, nodelist=nodes)
        degs = np.sum(A, axis=1)
        D_e = np.diag(np.power(degs, e))
        M = alpha * (D_e @ A @ D_e) + (1 - alpha) * np.eye(len(nodes))
        for _ in range(int(k)):
            Y = M @ Y
        labels = np.argmax(Y, axis=1)
        if return_y:
            return {node: int(labels[node_idx[node]]) for node in nodes}, Y
        return {node: int(labels[node_idx[node]]) for node in nodes}
    elif method == 'custom' and custom_code:
        local_env = {"G": G, "nx": nx, "np": np, "nodes": nodes, "Y": Y}
        try:
            exec(custom_code, {}, local_env)
            labels = local_env.get("labels")
            if labels is None:
                raise ValueError("Custom clustering code must define 'labels'")
            labels = np.asarray(labels).astype(int)
            cluster_map = {node: int(labels[node_idx[node]]) for node in nodes}
            Y_out = np.asarray(local_env.get("Y", Y))
            if return_y:
                return cluster_map, Y_out
            return cluster_map
        except Exception as exc:
            traceback.print_exc()
            raise CustomClusteringError(f"Failed to execute custom clustering code: {exc}") from exc
    # fallback: all nodes in one cluster
    if return_y:
        return {n: 0 for n in nodes}, Y
    return {n: 0 for n in nodes}

def graph_visualization_layout(graph_data, cluster_labels=None):
    import plotly.graph_objs as go
    import plotly.express as px
    import networkx as nx
    import numpy as np
    nodes = graph_data.get('nodes', {})
    edges = graph_data.get('edges', [])
    G = nx.Graph()
    for n in nodes:
        G.add_node(n)
    for u, v in edges:
        G.add_edge(u, v)

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    num_components = nx.number_connected_components(G) if num_nodes > 0 else 0
    diameter = nx.diameter(G) if num_nodes > 0 and nx.is_connected(G) else "N/A"
    avg_degree = (2 * num_edges / num_nodes) if num_nodes > 0 else 0
    avg_clustering = nx.average_clustering(G) if num_nodes > 0 else 0
    density = nx.density(G) if num_nodes > 0 else 0

    # Cluster color palette
    cluster_colors = px.colors.qualitative.Plotly
    node_cluster_colors = None
    node_text = [f"{n}" for n in G.nodes()]
    if cluster_labels:
        node_cluster_colors = [cluster_colors[cluster_labels.get(n, 0) % len(cluster_colors)] for n in G.nodes()]
        node_text = [f"{n} (C{cluster_labels.get(n, 0)})" for n in G.nodes()]

    return html.Div(style={"display": "flex", "flexDirection": "column", "width":"100%"},
        children=[
        html.H3("Graph Visualization"),
        html.Div(
            style={"display": "flex", "flexDirection": "row"},
            children=[
                html.Div(
                    dcc.Graph(id="graph", style={"height": "30vh"}),
                    style={"width": "80%"}
                ),
                html.Div([
                    html.H4("Graph Stats"),
                    html.Table([
                        html.Tr([html.Th("Property"), html.Th("Value")]),
                        html.Tr([html.Td("Number of nodes"), html.Td(num_nodes)]),
                        html.Tr([html.Td("Number of edges"), html.Td(num_edges)]),
                        html.Tr([html.Td("Connected components"), html.Td(num_components)]),
                        html.Tr([html.Td("Diameter"), html.Td(diameter)]),
                        html.Tr([html.Td("Average degree"), html.Td(f"{avg_degree:.2f}")]),
                        html.Tr([html.Td("Average clustering"), html.Td(f"{avg_clustering:.2f}")]),
                        html.Tr([html.Td("Density"), html.Td(f"{density:.3f}")]),
                    ], style={"marginTop": "10px", "marginBottom": "10px", "border": "1px solid #ccc", "width": "100%"})
                ], style={"width": "20%", "paddingLeft": "20px"})
            ]
        ),
        html.H3("Eigen Decomposition"),
        html.Div(style={"display": "flex", "justifyContent": "space-between"}, children=[
            dcc.Graph(id="matrix-l", style={"width": "48%", "height": "40vh"}),
            dcc.Graph(id="matrix-u", style={"width": "48%", "height": "40vh"})
        ]),
        dcc.Graph(id="matrix-s", style={"height": "15vh"}),
        html.Div(style={"display": "flex", "justifyContent": "space-between"}, children=[
            dcc.Graph(id="matrix-l-stack", style={"width": "48%", "height": "40vh"}),
            dcc.Graph(id="matrix-l-agg", style={"width": "48%", "height": "40vh"})
        ])
    ])

def settings_controls(include_clustering=False):
    controls = [graph_generator_controls()]
    if include_clustering:
        controls.append(clustering_controls())
    controls += [eigen_decomposition_controls(), html.Div(id="ek-pairs-table-panel", style={"marginBottom": "20px"}), node_edge_controls()]
    return html.Div([
        html.H2("Settings"),
        *controls
    ], style={"width": "30%", "overflowY": "auto", "height": "100vh", "padding": "10px", "borderLeft": "1px solid #ccc"})

def laplacian_hyperparameters_controls():
    return html.Div([
        html.H3("Laplacian Hyperparameters (L = D^E A D^E)^K"),
        html.Label("E (float):"),
        dcc.Input(id="e-slider", type="number", value=-0.5, step=0.1), html.Br(), html.Br(),
        html.Label("K (int ≥1):"),
        dcc.Input(id="k-slider", type="number", value=1, step=1), html.Br(), html.Br(),
    ], style={"border": "1px solid #ccc", "padding": "10px", "margin-bottom": "10px"})

def ek_pairs_controls():
    return html.Div([
        html.H3("(E, K) Pairs for Aggregation/Stacking"),
        html.Div("The lists below generate all (E,K) pairs for the bottom two figures.", style={"fontSize": "0.9em", "color": "#555"}),
        html.Label("List of E values (comma-separated, e.g. -0.5,0,0.5):"),
        dcc.Input(id="eps-list", type="text", value="-0.5, -0.4, -0.3"), html.Br(), html.Br(),
        html.Label("List of K values (comma-separated, e.g. 1,2,3,4):"),
        dcc.Input(id="ks-list", type="text", value="1,2,3,4"), html.Br(), html.Br(),
        html.Label("Aggregation:"),
        dcc.Dropdown(
            id="agg-method",
            options=[
                {"label": "Mean", "value": "mean"},
                {"label": "Sum", "value": "sum"},
                {"label": "Max", "value": "max"},
                {"label": "Min", "value": "min"}
            ],
            value="mean"
        ), html.Br(),
        html.Label("Aggregation Dimension:"),
        dcc.Dropdown(
            id="agg-dim",
            options=[
                {"label": "Axis 0 (rows)", "value": 0},
                {"label": "Axis 1 (columns)", "value": 1},
                {"label": "Axis 2 (pairs)", "value": 2}
            ],
            value=1
        )
    ], style={"border": "1px solid #ccc", "padding": "10px", "margin-bottom": "10px"})

def eigen_decomposition_controls():
    # Always include both e-slider and e-input, but hide the one not needed for the current tab
    return html.Div([
        html.Label("Exponent (E):"),
        dcc.Slider(id="e-slider", min=-2, max=2, step=0.1, value=-0.5, 
                   marks={-2: '-2', -1: '-1', 0: '0', 1: '1', 2: '2'},
                   tooltip={"placement": "bottom", "always_visible": False},
                   style={"marginBottom": "20px", "display": "block"},
        ),
        dcc.Input(id="e-input", type="number", value=-0.5, step=0.1, 
                  style={"marginBottom": "20px", "display": "none"}),
        # For backward compatibility, but now split into two panels
        ek_pairs_controls()
    ])

def node_edge_controls():
    return html.Div([
        html.H3("Nodes & Edges"),
        html.Label("Node ID:"),
        dcc.Input(id="node-id", type="text", value="1"), html.Br(),
        html.Button("Add / Update Node", id="add-node", n_clicks=0), html.Br(), html.Br(),
        html.Label("Remove Node ID:"), 
        dcc.Input(id="remove-node-id", type="text", value="1"),
        html.Button("Remove Node", id="remove-node", n_clicks=0), html.Br(), html.Br(),
        html.Label("Add Edge (src → tgt):"), html.Br(),
        dcc.Input(id="edge-source", type="text", value="1"),
        dcc.Input(id="edge-target", type="text", value="2"),
        html.Button("Add Edge", id="add-edge", n_clicks=0), html.Br(), html.Br(),
        html.Label("Remove Edge (src → tgt):"), html.Br(),
        dcc.Input(id="remove-edge-source", type="text", value="1"),
        dcc.Input(id="remove-edge-target", type="text", value="2"),
        html.Button("Remove Edge", id="remove-edge", n_clicks=0)
    ], style={"border": "1px solid #ccc", "padding": "10px", "margin-bottom": "20px"})

def graph_generator_controls():
    return html.Div([
        html.H3("Random Graph Generator"),
        html.Label("Graph Family:"),
        dcc.Dropdown(
            id="rg-graph-family",
            options=[
                {"label": "Erdős–Rényi", "value": "ER"},
                {"label": "Barabási–Albert", "value": "BA"},
                {"label": "Watts–Strogatz", "value": "WS"},
                {"label": "Scale Free", "value": "SF"},
                {"label": "Star", "value": "Star"},
                {"label": "Lattice", "value": "Lattice"},
                {"label": "Delaunay Triangles", "value": "Delaunay"}
            ],
            value="SF"
        ),
        html.Br(),
        html.Label("Number of Nodes:"),
        dcc.Input(id="rg-num-nodes", type="number", value=10, min=1),
        html.Br(), html.Br(),
        html.Label("Density / Probability (0-1, for ER/WS):"),
        dcc.Slider(id="rg-density", min=0, max=1, step=0.01, value=0.3, marks={0: "0", 0.5: "0.5", 1: "1"}),
        html.Br(),
        html.Button("Generate Graph", id="generate-graph", n_clicks=0),
        html.Button("Clear Graph", id="clear-graph", n_clicks=0)
    ], style={"border": "1px solid #ccc", "padding": "10px", "margin-bottom": "20px"})

def clustering_controls(tab=None):
    # Always include alpha-slider, but hide unless tab == 'tab-clustering'
    alpha_style = {"display": "block" if tab == 'tab-clustering' else "none", "marginBottom": "20px"}
    return html.Div([
        html.H3("Clustering Controls"),
        html.Label("Clustering Method:"),
        dcc.Dropdown(
            id="clustering-method",
            options=[{"label": "Spectral LPA", "value": "spectral_lpa"}],
            value="spectral_lpa",
            disabled=True
        ),
        html.Br(),
        html.Label("Alpha (Weighting parameter, α):"),
        html.Div(
            dcc.Slider(id="alpha-slider", min=0, max=1, step=0.05, value=0.5,
                       marks={0: '0', 0.5: '0.5', 1: '1'}),
            style=alpha_style
        ),
        html.Br(),
        html.Label("Exponent (E):"),
        dcc.Slider(id="e-slider", min=-2, max=2, step=0.1, value=-0.5,
                   marks={-2: '-2', -1: '-1', 0: '0', 1: '1', 2: '2'},
                   tooltip={"placement": "bottom", "always_visible": False}),
        html.Label("Iterations (K):"),
        dcc.Slider(id="k-slider", min=1, max=20, step=1, value=5,
                   marks={i: str(i) for i in range(1, 21)},
                   tooltip={"placement": "bottom", "always_visible": False})
    ], style={"border": "1px solid #ccc", "padding": "10px", "margin-bottom": "20px"})

def eigen_controls_with_tab(tab):
    # Deprecated: do not use, controls are now always present in the root layout
    return None
