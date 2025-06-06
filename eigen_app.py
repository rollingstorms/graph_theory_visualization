import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
import numpy as np
from scipy.spatial import Delaunay

# Import documentation content
from eigen_documentation import DOCS_CONTENT


# -------------------------
# Graph Data Helper
# -------------------------
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

# -------------------------
# Dash App Setup
# -------------------------
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

app.layout = html.Div(
    style={"display": "flex", "height": "100vh"},
    children=[
        # Left Column: Tabs for Graph Tools and Documentation
        html.Div([
            dcc.Tabs(id="tabs", value='tab-graph', children=[
                dcc.Tab(label='Graph Tools', value='tab-graph'),
                dcc.Tab(label='Documentation', value='tab-docs')
            ]),
            html.Div(id='tab-content')
        ],
        style={"width": "70%", "padding": "10px", "overflowY": "auto", "height": "100vh"}
        ),
        # Right Column: Controls and Graph Stats
        html.Div(
            children=[
                html.H2("Settings"),
                # Random Graph Generator
                html.Div([
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
                        value="SF",  # Default to Scale Free
                    ), html.Br(),
                    html.Label("Number of Nodes:"), dcc.Input(id="rg-num-nodes", type="number", value=10, min=1), html.Br(), html.Br(),
                    html.Label("Density / Probability (0-1, for ER/WS):"),
                    dcc.Slider(id="rg-density", min=0, max=1, step=0.01, value=0.3, marks={0: "0", 0.5: "0.5", 1: "1"}), html.Br(),
                    html.Button("Generate Graph", id="generate-graph", n_clicks=0),
                    html.Button("Clear Graph", id="clear-graph", n_clicks=0)
                ], style={"border": "1px solid #ccc", "padding": "10px", "margin-bottom": "20px"}),
                # Eigen Decomposition Controls
                html.Div([
                    html.H3("Hyperparameters for L = (D^E A D^E) ^ K"),
                    html.Div("The E and K below affect the top two figures (L, U). The lists below generate all (E,K) pairs for the bottom two figures.", style={"fontSize": "0.9em", "color": "#555"}),
                    html.Label("E (float):"),
                    dcc.Input(id="e-slider", type="number", value=-0.5, step=0.1), html.Br(), html.Br(),
                    html.Label("K (int ≥1):"),
                    dcc.Input(id="k-slider", type="number", value=1, step=1), html.Br(), html.Br(),
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
                ], style={"border": "1px solid #ccc", "padding": "10px", "margin-bottom": "10px"}),
                # E, K Pairs Table (below hyperparameter panel)
                html.Div(id="ek-pairs-table-panel", style={"marginBottom": "20px"}),
                # Node & Edge Controls
                html.Div([
                    html.H3("Nodes & Edges"),
                    html.Label("Node ID:"),
                    dcc.Input(id="node-id", type="text", value="1"), html.Br(),
                    html.Button("Add / Update Node", id="add-node", n_clicks=0), html.Br(), html.Br(),
                    html.Label("Remove Node ID:"), dcc.Input(id="remove-node-id", type="text", value="1"),
                    html.Button("Remove Node", id="remove-node", n_clicks=0), html.Br(), html.Br(),
                    html.Label("Add Edge (src → tgt):"), html.Br(),
                    dcc.Input(id="edge-source", type="text", value="1"),
                    dcc.Input(id="edge-target", type="text", value="2"),
                    html.Button("Add Edge", id="add-edge", n_clicks=0), html.Br(), html.Br(),
                    html.Label("Remove Edge (src → tgt):"), html.Br(),
                    dcc.Input(id="remove-edge-source", type="text", value="1"),
                    dcc.Input(id="remove-edge-target", type="text", value="2"),
                    html.Button("Remove Edge", id="remove-edge", n_clicks=0)
                ], style={"border": "1px solid #ccc", "padding": "10px", "margin-bottom": "20px"}),
                # Layout Mode
                # (Remove this block entirely)
            ],
            style={"width": "30%", "overflowY": "auto", "height": "100vh", "padding": "10px", "borderLeft": "1px solid #ccc"}
        ),
        dcc.Store(id="graph-store", data=default_graph())
    ]
)

# -------------------------
# Callbacks
# -------------------------
@app.callback(
    Output('tab-content', 'children'),
    Input('tabs', 'value'),
    Input('graph-store', 'data')
)
def render_tab_content(tab, graph_data):
    """Render the content for the selected tab."""
    if tab == 'tab-graph':
        # Compute graph stats for the table
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

        return html.Div([
            html.H3("Graph Visualization"),
            html.Div(
                style={"display": "flex", "flexDirection": "row"},
                children=[
                    html.Div(
                        dcc.Graph(id="graph", style={"height": "30vh"}),
                        style={"width": "80%"}
                    ),
                    html.Div(
                        [
                            html.H4("Graph Stats"),
                            html.Table([
                                html.Tr([html.Th("Property"), html.Th("Value")]),
                                html.Tr([html.Td("Number of nodes"), html.Td(num_nodes)]),
                                html.Tr([html.Td("Number of edges"), html.Td(num_edges)]),
                                html.Tr([html.Td("Connected components"), html.Td(num_components)]),
                                html.Tr([html.Td("Diameter"), html.Td(diameter)]),
                                html.Tr([html.Td("Average degree"), html.Td(f"{avg_degree:.2f}")]),
                                html.Tr([html.Td("Density"), html.Td(f"{density:.3f}")]),
                            ], style={"marginTop": "10px", "marginBottom": "10px", "border": "1px solid #ccc", "width": "100%"}),
                        ],
                        style={"width": "20%", "paddingLeft": "20px"}
                    )
                ]
            ),
            html.H3("Eigen Decomposition"),
            html.Div(
                style={"display": "flex", "justifyContent": "space-between"},
                children=[
                    dcc.Graph(id="matrix-l", style={"width": "48%", "height": "40vh"}),
                    dcc.Graph(id="matrix-u", style={"width": "48%", "height": "40vh"})
                ]
            ),
            dcc.Graph(id="matrix-s", style={"height": "15vh"}),
            html.Div(
                style={"display": "flex", "justifyContent": "space-between"},
                children=[
                    dcc.Graph(id="matrix-l-stack", style={"width": "48%", "height": "40vh"}),
                    dcc.Graph(id="matrix-l-agg", style={"width": "48%", "height": "40vh"})
                ]
            ),
            # Remove ek-pairs-list from here
        ])
    elif tab == 'tab-docs':
        # Load documentation from eigen_documentation.py
        return html.Div(DOCS_CONTENT, style={"padding": "20px"})

        dcc.Store(id="graph-store", data=default_graph())
    


@app.callback(
    Output("graph-store", "data"),
    Input("add-node", "n_clicks"),
    Input("remove-node", "n_clicks"),
    Input("add-edge", "n_clicks"),
    Input("remove-edge", "n_clicks"),
    Input("generate-graph", "n_clicks"),
    Input("clear-graph", "n_clicks"),
    State("graph-store", "data"),
    State("node-id", "value"),
    # Remove these non-existent States:
    # State("node-x-slider", "value"), State("node-y-slider", "value"), State("node-z-slider", "value"), State("node-extra-slider", "value"),
    State("remove-node-id", "value"),
    State("edge-source", "value"),
    State("edge-target", "value"),
    State("remove-edge-source", "value"),
    State("remove-edge-target", "value"),
    State("rg-graph-family", "value"),
    State("rg-num-nodes", "value"),
    State("rg-density", "value"),
    prevent_initial_call=True
)
def update_graph_store(add_node_clicks, remove_node_clicks, add_edge_clicks, remove_edge_clicks,
                       gen_clicks, clear_clicks,
                       graph_data, node_id,
                       # Remove x, y, z, extra from args
                       remove_node_id, src, tgt,
                       rem_src, rem_tgt,
                       rg_family, rg_num_nodes, rg_density):
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
            # Ensure alpha+beta+gamma == 1 and gamma > 0
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

@app.callback(
    Output('graph', 'figure'),
    Input('graph-store', 'data'),
    Input('matrix-l', 'figure')
)
def update_graph_figure(graph_data, matrix_l_fig):
    """Draw the graph with node and edge coloring based on Laplacian values."""
    nodes = graph_data.get('nodes', {})
    edges = graph_data.get('edges', [])
    G = nx.Graph()
    for n, props in nodes.items():
        G.add_node(n, pos=(props['x'], props['y'], props['z']))
    for u, v in edges:
        if u in nodes and v in nodes:
            G.add_edge(u, v)

    # Always use spring2d layout
    spring_pos = nx.spring_layout(G, iterations=50)
    pos2d = {n: (xy[0], xy[1]) for n, xy in spring_pos.items()}

    # --- Edge and Node coloring based on Laplacian (matrix-l) values ---
    node_colors = None
    edge_colors = None
    colormap = px.colors.sequential.Plasma  # Unified colormap for all

    if matrix_l_fig and "data" in matrix_l_fig and len(matrix_l_fig["data"]) > 0:
        z = matrix_l_fig["data"][0].get("z")
        if z is not None:
            try:
                # z_arr may be a list of dicts (from plotly px.imshow)
                if isinstance(z, dict):
                    z = z['_inputArray']
                    z_arr = np.array([list(row.values()) for row in z])
                else:
                    z_arr = np.array(z)
                # Store diagonal for node coloring
                node_colors = np.diag(z_arr)
                node_list = list(G.nodes())
                edge_colors = []
                for u, v in G.edges():
                    try:
                        i = node_list.index(u)
                        j = node_list.index(v)
                        val = z_arr[i, j]
                    except Exception:
                        val = 0
                    edge_colors.append(val)
            except Exception:
                node_colors = None
                edge_colors = None

    # Fallbacks if not available
    if node_colors is None or len(nodes) == 0 or len(node_colors) != len(nodes):
        node_colors = "LightSkyBlue"
    if edge_colors is None or len(edge_colors) != G.number_of_edges():
        edge_colors = '#888'

    # Normalize edge colors for colormap if numeric
    use_edge_cmap = isinstance(edge_colors, (list, np.ndarray)) and not isinstance(edge_colors, str)
    if use_edge_cmap:
        edge_color_vals = np.array(edge_colors)
        if np.ptp(edge_color_vals) == 0:
            edge_color_vals = np.zeros_like(edge_color_vals)
        edge_color_norm = (edge_color_vals - np.min(edge_color_vals)) / (np.ptp(edge_color_vals) + 1e-9)
        edge_colors_mapped = [px.colors.sample_colorscale(colormap, v)[0] for v in edge_color_norm]
    else:
        edge_colors_mapped = edge_colors if isinstance(edge_colors, list) else [edge_colors]*G.number_of_edges()

    # Normalize node colors for colormap if numeric
    use_node_cmap = isinstance(node_colors, (list, np.ndarray)) and not isinstance(node_colors, str)
    if use_node_cmap:
        node_color_vals = np.array(node_colors)
        if np.ptp(node_color_vals) == 0:
            node_color_vals = np.zeros_like(node_color_vals)
        node_color_norm = (node_color_vals - np.min(node_color_vals)) / (np.ptp(node_color_vals) + 1e-9)
        node_colors_mapped = [px.colors.sample_colorscale(colormap, v)[0] for v in node_color_norm]
    else:
        node_colors_mapped = node_colors if isinstance(node_colors, list) else [node_colors]*len(G.nodes())

    # Plot each edge as a separate trace to allow per-edge color
    edge_traces = []
    edge_color_vals = []
    for idx, (u, v) in enumerate(G.edges()):
        x0, y0 = pos2d[u]
        x1, y1 = pos2d[v]
        color = edge_colors_mapped[idx] if use_edge_cmap else edge_colors_mapped[idx]
        hoverlabel = f"Edge: {u} → {v}<br>L[{u},{v}] = {edge_colors[idx]:.3g}" if use_edge_cmap else f"Edge: {u} → {v}"
        xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
        # Store color value for colorbar
        if use_edge_cmap:
            edge_color_vals.append(edge_colors[idx])
        edge_traces.append(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode='lines',
            line=dict(color=color, width=4),
            hoverinfo='skip',
            showlegend=False
        ))
        edge_traces.append(go.Scatter(
            x=[xm], y=[ym],
            mode='markers',
            marker=dict(size=12, color=color, opacity=0.7, symbol='circle'),
            hoverinfo='text',
            hovertext=hoverlabel,
            showlegend=False
        ))

    # Add a colorbar for edges if using colormap and there are edges
    colorbar_trace = None
    if use_edge_cmap and len(edge_color_vals) > 0:
        # Add a dummy scatter for colorbar
        colorbar_trace = go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(
                colorscale=colormap,
                cmin=min(edge_color_vals),
                cmax=max(edge_color_vals),
                colorbar=dict(title="Edge L value"),
                color=edge_color_vals,
                size=0.1  # invisible
            ),
            hoverinfo='none',
            showlegend=False
        )

    node_x = [pos2d[n][0] for n in G.nodes()]
    node_y = [pos2d[n][1] for n in G.nodes()]
    hover_text = []
    for idx, n in enumerate(G.nodes()):
        deg = G.degree[n]
        incident_edges = [f"{u}→{v}" if u == n else f"{v}→{u}" for u, v in G.edges(n)]
        edge_str = ", ".join(incident_edges) if incident_edges else "None"
        node_val = node_colors[idx] if use_node_cmap else ""
        hover_text.append(f"Node: {n}<br>Degree: {deg}<br>Edges: {edge_str}" + (f"<br>L[{n},{n}] = {node_val:.3g}" if use_node_cmap else ""))

    text = [f"{n}" for n in G.nodes()]
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=text,
        textposition='bottom center',
        marker=dict(
            size=12,
            color=node_colors_mapped,
            colorscale=colormap if use_node_cmap else None,
            colorbar=dict(title="Laplacian Diag") if use_node_cmap else None,
            line=dict(width=2)
        ),
        hoverinfo='text',
        hovertext=hover_text,
        showlegend=False  # Hide node trace from legend
    )

    title = None
    fig = go.Figure(
        data=edge_traces + ([colorbar_trace] if colorbar_trace else []) + [node_trace],
        layout=go.Layout(
            title=title,
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )
    return fig

@app.callback(
    Output('matrix-l', 'figure'),
    Output('matrix-u', 'figure'),
    Output('matrix-s', 'figure'),
    Output('matrix-l-stack', 'figure'),
    Output('matrix-l-agg', 'figure'),
    # Remove Output('ek-pairs-list', 'children'),
    Input('graph-store', 'data'),
    Input('e-slider', 'value'),
    Input('k-slider', 'value'),
    Input('eps-list', 'value'),
    Input('ks-list', 'value'),
    Input('agg-method', 'value'),
    Input('agg-dim', 'value')
)
def update_eigen(graph_data, e, k, eps_list, ks_list, agg_method, agg_dim):
    """Compute and visualize Laplacian and its spectral decomposition for the current graph."""
    nodes = list(graph_data.get('nodes', {}).keys())
    n = len(nodes)
    if n == 0:
        return go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure()
    A = np.zeros((n,n))
    for u,v in graph_data.get('edges', []):
        if u in nodes and v in nodes and u != v:  # Prevent self-loops in adjacency
            i,j = nodes.index(u), nodes.index(v)
            A[i,j] = 1; A[j,i] = 1

    # --- Single L and U as before ---
    degs = A.sum(axis=1)
    D_e = np.diag(degs**e)
    k = max(k, 1)
    
    L = D_e @ A @ D_e
    L = np.linalg.matrix_power(L, int(k)) if k == int(k) else L
    U, S, Vt = np.linalg.svd(L)
    figL = px.imshow(L, aspect='equal')
    figL.update_layout(title=f"L = D^{e} A^{k} D^{e}", margin=dict(l=20,r=20,t=30,b=20))
    figL.update_xaxes(title_text='Nodes', ticktext=nodes)
    figL.update_yaxes(title_text='Nodes', ticktext=nodes)
    figU = px.imshow(U, aspect='equal')
    figU.update_layout(title='U (SVD)', margin=dict(l=20,r=20,t=30,b=20))
    figU.update_yaxes(title_text='Nodes', ticktext=nodes)
    figU.update_xaxes(title_text='Eigenvectors', ticktext=[f"v{i+1}" for i in range(U.shape[1])])
    # New: SVD eigenvalues plot
    figS = go.Figure(go.Bar(x=[f"v{i+1}" for i in range(len(S))], y=S))
    figS.update_layout(title="Eigenvalues (S)", xaxis_title="Eigenvector", yaxis_title="Value", margin=dict(l=20,r=20,t=30,b=20))

    # --- Parse (e,k) pairs from eps_list and ks_list ---
    eps = []
    ks = []
    if eps_list:
        try:
            eps = [float(x.strip()) for x in eps_list.split(',') if x.strip() != ""]
        except Exception:
            eps = []
    if ks_list:
        try:
            ks = [float(x.strip()) for x in ks_list.split(',') if x.strip() != ""]
        except Exception:
            ks = []
    ek_list = [(e_i, k_j) for k_j in ks for e_i in eps] if eps and ks else [(e, k)]

    # --- Stack Ls ---
    L_stack = []
    for e_i, k_i in ek_list:
        D_ei = np.diag(degs**e_i)
        k_i = max(k_i, 1)
        L_i = D_ei @ A @ D_ei
        L_i = np.linalg.matrix_power(L_i, int(k_i)) if k_i == int(k_i) else L_i
        L_stack.append(L_i)
    L_stack = np.stack(L_stack, axis=-1)  # shape (n, n, num_pairs)

    # --- 3D Prism Plot ---
    # We'll plot as transparent colored cubes using go.Volume
    # For visualization, we threshold small values for transparency
    opacity = 0.3
    figStack = go.Figure()
    for idx in range(L_stack.shape[2]):
        figStack.add_trace(go.Surface(
            z=np.full_like(L_stack[:,:,idx], idx),
            x=np.arange(n),
            y=np.arange(n),
            surfacecolor=L_stack[:,:,idx],
            colorscale='Plasma',
            opacity=opacity,
            showscale=(idx==0),
            name=f"L({ek_list[idx][0]},{ek_list[idx][1]})"
        ))
    figStack.update_layout(
        title="Stacked L Matrices (n x n x num_pairs)",
        scene=dict(
            xaxis_title="Node i",
            yaxis_title="Node j",
            zaxis_title="Pair Index"
        ),
        margin=dict(l=20,r=20,t=30,b=20)
    )

    # --- Aggregation ---
    axis = agg_dim if agg_dim is not None else 2
    if agg_method == "mean":
        L_agg = np.mean(L_stack, axis=axis)
    elif agg_method == "sum":
        L_agg = np.sum(L_stack, axis=axis)
    elif agg_method == "max":
        L_agg = np.max(L_stack, axis=axis)
    elif agg_method == "min":
        L_agg = np.min(L_stack, axis=axis)
    else:
        L_agg = np.mean(L_stack, axis=axis)

    # Set axis labels for the aggregated figure
    if axis == 0:
        x_title = 'Nodes'
        y_title = 'Pairs'
        y_ticktext = [f"({e},{k})" for (e, k) in ek_list]
        x_ticktext = nodes
        aspect = 'auto'
    elif axis == 1:
        x_title = 'Pairs'
        y_title = 'Nodes'
        x_ticktext = [f"({e},{k})" for (e, k) in ek_list]
        y_ticktext = nodes
        aspect = 'auto'
    else:  # axis == 2
        x_title = 'Nodes'
        y_title = 'Nodes'
        x_ticktext = nodes
        y_ticktext = nodes
        aspect = 'equal'

    figAgg = px.imshow(L_agg, aspect=aspect)
    figAgg.update_layout(title=f"Aggregated L ({agg_method}, axis={axis})", margin=dict(l=20,r=20,t=30,b=20))
    figAgg.update_xaxes(title_text=x_title, ticktext=x_ticktext)
    figAgg.update_yaxes(title_text=y_title, ticktext=y_ticktext)

    return figL, figU, figS, figStack, figAgg

@app.callback(
    Output('ek-pairs-table-panel', 'children'),
    Input('e-slider', 'value'),
    Input('k-slider', 'value'),
    Input('eps-list', 'value'),
    Input('ks-list', 'value')
)
def render_ek_pairs_table(e, k, eps_list, ks_list):
    """Display the E, K pairs as a table below the hyperparameter panel."""
    # Parse (e,k) pairs
    eps = []
    ks = []
    if eps_list:
        try:
            eps = [float(x.strip()) for x in eps_list.split(',') if x.strip() != ""]
        except Exception:
            eps = []
    if ks_list:
        try:
            ks = [float(x.strip()) for x in ks_list.split(',') if x.strip() != ""]
        except Exception:
            ks = []
    if eps and ks:
        ek_list = [(e_i, k_j) for k_j in ks for e_i in eps]
    else:
        ek_list = [(e, k)]
    if not ek_list:
        return html.Div("No (E, K) pairs.")
    return html.Table(
        [html.Tr([html.Th("E"), html.Th("K")])] +
        [html.Tr([html.Td(str(e)), html.Td(str(k))]) for (e, k) in ek_list],
        style={"marginTop": "10px", "marginBottom": "10px", "border": "1px solid #ccc", "width": "100%"}
    )

if __name__ == '__main__':
    app.run(debug=True)