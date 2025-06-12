from dash import Input, Output, State, callback_context
import numpy as np
import networkx as nx
from graph_utils import render_tab_content, update_graph_store
import plotly.graph_objs as go
import plotly.express as px
import dash


def register_callbacks(app):
    @app.callback(
        Output('tab-content', 'children'),
        Input('tabs', 'value'),
        Input('graph-tools-store', 'data'),
        Input('clustering-store', 'data')
    )
    def _render_tab_content(tab, graph_data, clustering_store):
        clustering_method = clustering_store.get('clustering_method', 'lpa') if clustering_store else 'lpa'
        clustering_step = clustering_store.get('clustering_step', 0) if clustering_store else 0
        clustering_node_order = clustering_store.get('clustering_node_order', '') if clustering_store else ''
        custom_code = clustering_store.get('custom_code') if clustering_store else None
        return render_tab_content(tab, graph_data, clustering_method, clustering_step, clustering_node_order, custom_code)

    # --- Unified graph figure callback for both tabs ---
    @app.callback(
        Output('graph', 'figure'),
        [Input('tabs', 'value'),
         Input('graph-tools-store', 'data'),
         Input('clustering-store', 'data'),
         Input('e-input', 'value'),
         Input('k-slider', 'value')],
        prevent_initial_call=False
    )
    def update_graph_figure(tab, graph_tools_data, clustering_data, e_value, k_value):
        import traceback
        try:
            # Ensure stores are always dicts, never None
            if graph_tools_data is None:
                from graph_utils import empty_graph
                graph_tools_data = empty_graph()
            if clustering_data is None:
                from graph_utils import empty_graph
                clustering_data = empty_graph()
            if tab == 'tab-graph':
                nodes = graph_tools_data.get('nodes', {})
                edges = graph_tools_data.get('edges', [])
                G = nx.Graph()
                for n, props in nodes.items():
                    G.add_node(n, pos=(props['x'], props['y'], props.get('z', 0.0)))
                for u, v in edges:
                    if u in nodes and v in nodes:
                        G.add_edge(u, v)
                # Use stored positions for visualization
                pos2d = {n: (props['x'], props['y']) for n, props in nodes.items()}
                edge_colors = None
                colormap = px.colors.sequential.Plasma
                use_edge_cmap = False
                # Compute Laplacian diagonal for node coloring
                node_colors = "LightSkyBlue"
                use_node_cmap = False
                if len(nodes) > 0:
                    node_list = list(nodes.keys())
                    index = {n: i for i, n in enumerate(node_list)}
                    A = np.zeros((len(node_list), len(node_list)))
                    for u, v in edges:
                        if u in index and v in index and u != v:
                            i, j = index[u], index[v]
                            A[i, j] = 1
                            A[j, i] = 1
                    degs = A.sum(axis=1)
                    e = e_value if e_value is not None else -0.5
                    k_val = int(k_value) if k_value is not None else 1
                    D_e = np.diag(np.power(degs, e))
                    L = D_e @ A @ D_e
                    L = np.linalg.matrix_power(L, k_val) if k_val == int(k_val) else L
                    node_colors = np.diag(L)
                    use_node_cmap = True
                if node_colors is None or len(nodes) == 0 or (use_node_cmap and len(node_colors) != len(nodes)):
                    node_colors = "LightSkyBlue"
                if edge_colors is None or (use_edge_cmap and len(edge_colors) != G.number_of_edges()):
                    edge_colors = '#888'
                if use_edge_cmap:
                    edge_color_vals = np.array(edge_colors)
                    if np.ptp(edge_color_vals) == 0:
                        edge_color_vals = np.zeros_like(edge_color_vals)
                    edge_color_norm = (edge_color_vals - np.min(edge_color_vals)) / (np.ptp(edge_color_vals) + 1e-9)
                    edge_colors_mapped = [px.colors.sample_colorscale(colormap, v)[0] for v in edge_color_norm]
                else:
                    edge_colors_mapped = edge_colors if isinstance(edge_colors, list) else [edge_colors]*G.number_of_edges()
                # When mapping node colors, ensure node_colors is numeric
                node_color_vals = node_colors if use_node_cmap and node_colors is not None else np.zeros(len(G.nodes()))
                # Defensive: if node_color_vals is a dict, extract array or fallback
                if isinstance(node_color_vals, dict):
                    if '_inputArray' in node_color_vals:
                        node_color_vals = np.array(node_color_vals['_inputArray'])
                    elif 'flat' in node_color_vals:
                        node_color_vals = np.array(node_color_vals['flat'])
                    else:
                        arr = next((v for v in node_color_vals.values() if isinstance(v, (list, np.ndarray))), None)
                        if arr is not None:
                            node_color_vals = np.array(arr)
                        else:
                            node_color_vals = np.zeros(len(G.nodes()))
                # Final fallback: if still not array-like, fallback
                try:
                    node_color_vals = np.asarray(node_color_vals, dtype=float).flatten()
                except Exception:
                    node_color_vals = np.zeros(len(G.nodes()))
                # Now safe to use np.ptp(node_color_vals)
                if node_color_vals.size == 0 or np.ptp(node_color_vals) == 0:
                    node_colors_mapped = [colormap[0]] * len(G.nodes())
                else:
                    node_colors_mapped = [px.colors.sample_colorscale(colormap, (v - np.min(node_color_vals)) / np.ptp(node_color_vals))[0] for v in node_color_vals]
                edge_traces = []
                edge_color_vals = []
                for idx, (u, v) in enumerate(G.edges()):
                    x0, y0 = pos2d[u]
                    x1, y1 = pos2d[v]
                    color = edge_colors_mapped[idx] if use_edge_cmap else edge_colors_mapped[idx]
                    hoverlabel = f"Edge: {u} → {v}" if not use_edge_cmap else f"Edge: {u} → {v}<br>L[{u},{v}] = {edge_colors[idx]:.3g}"
                    xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
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
                colorbar_trace = None
                if use_edge_cmap and len(edge_color_vals) > 0:
                    colorbar_trace = go.Scatter(
                        x=[None], y=[None],
                        mode='markers',
                        marker=dict(
                            colorscale=colormap,
                            cmin=min(edge_color_vals),
                            cmax=max(edge_color_vals),
                            colorbar=dict(title="Edge L value"),
                            color=edge_color_vals,
                            size=0.1
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
                    # Fix: only format node_val as float if it's a number
                    if use_node_cmap:
                        try:
                            val_str = f"{float(node_val):.3g}"
                        except Exception:
                            val_str = str(node_val)
                        hover_text.append(f"Node: {n}<br>Degree: {deg}<br>Edges: {edge_str}<br>L[{n},{n}] = {val_str}")
                    else:
                        hover_text.append(f"Node: {n}<br>Degree: {deg}<br>Edges: {edge_str}")
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
                    showlegend=False
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
            elif tab == 'tab-clustering':
                # Laplacian values are not required for clustering tab visualization
                nodes = clustering_data.get('nodes', {})
                edges = clustering_data.get('edges', [])
                G = nx.Graph()
                for n, props in nodes.items():
                    G.add_node(n, pos=(props['x'], props['y'], props.get('z', 0.0)))
                for u, v in edges:
                    if u in nodes and v in nodes:
                        G.add_edge(u, v)
                # Use stored positions for visualization
                pos2d = {n: (props['x'], props['y']) for n, props in nodes.items()}
                cluster_labels = clustering_data.get('cluster_labels')
                cluster_colors = px.colors.qualitative.Plotly
                node_colors = [cluster_colors[cluster_labels.get(n, 0) % len(cluster_colors)] for n in G.nodes()]
                edge_colors = '#888'
                node_colors_mapped = node_colors
                edge_colors_mapped = [edge_colors]*G.number_of_edges()
                edge_traces = []
                for idx, (u, v) in enumerate(G.edges()):
                    x0, y0 = pos2d[u]
                    x1, y1 = pos2d[v]
                    color = edge_colors_mapped[idx]
                    hoverlabel = f"Edge: {u} → {v}"
                    xm, ym = (x0 + x1) / 2, (y0 + y1) / 2
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
                node_x = [pos2d[n][0] for n in G.nodes()]
                node_y = [pos2d[n][1] for n in G.nodes()]
                hover_text = []
                for idx, n in enumerate(G.nodes()):
                    deg = G.degree[n]
                    incident_edges = [f"{u}→{v}" if u == n else f"{v}→{u}" for u, v in G.edges(n)]
                    edge_str = ", ".join(incident_edges) if incident_edges else "None"
                    hover_text.append(f"Node: {n}<br>Degree: {deg}<br>Edges: {edge_str}")
                text = [f"{n}" for n in G.nodes()]
                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text',
                    text=text,
                    textposition='bottom center',
                    marker=dict(
                        size=12,
                        color=node_colors_mapped,
                        line=dict(width=2)
                    ),
                    hoverinfo='text',
                    hovertext=hover_text,
                    showlegend=False
                )
                title = None
                fig = go.Figure(
                    data=edge_traces + [node_trace],
                    layout=go.Layout(
                        title=title,
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    )
                )
                return fig
            else:
                return dash.no_update
        except Exception as e:
            print("[Dash Callback Error] update_graph_figure:", e)
            traceback.print_exc()
            return dash.no_update

    # --- Unified store sync/update callback ---
    @app.callback(
        [Output('clustering-store', 'data'), Output('graph-tools-store', 'data')],
        [Input('tabs', 'value'),
         Input('add-node', 'n_clicks'),
         Input('remove-node', 'n_clicks'),
         Input('add-edge', 'n_clicks'),
         Input('remove-edge', 'n_clicks'),
         Input('generate-graph', 'n_clicks'),
        Input('clear-graph', 'n_clicks'),
        Input('alpha-slider', 'value'),
        Input('e-input', 'value'),  # changed from e-slider to e-input
        Input('k-slider', 'value'),
        Input('clustering-method', 'value'),
        Input('clustering-code', 'value')],
        [State('graph-tools-store', 'data'),
         State('clustering-store', 'data'),
         State('node-id', 'value'),
         State('remove-node-id', 'value'),
         State('edge-source', 'value'),
         State('edge-target', 'value'),
         State('remove-edge-source', 'value'),
         State('remove-edge-target', 'value'),
         State('rg-graph-family', 'value'),
         State('rg-num-nodes', 'value'),
         State('rg-density', 'value')],
        prevent_initial_call=False
    )
    def unified_store_update(tab, add_node, remove_node, add_edge, remove_edge, generate_graph, clear_graph, alpha, e, k, clustering_method, clustering_code,
                            graph_tools_data, clustering_data,
                            node_id, remove_node_id, edge_source, edge_target, remove_edge_source, remove_edge_target,
                            rg_graph_family, rg_num_nodes, rg_density):
        # Ensure stores are always dicts, never None
        if graph_tools_data is None:
            from graph_utils import default_graph
            graph_tools_data = default_graph()
        if clustering_data is None:
            from graph_utils import default_graph
            clustering_data = default_graph()
        # Defensive: always ensure dicts
        if not isinstance(graph_tools_data, dict):
            from graph_utils import default_graph
            graph_tools_data = default_graph()
        if not isinstance(clustering_data, dict):
            from graph_utils import default_graph
            clustering_data = default_graph()
        # Graph Tools tab: update graph-tools-store
        if tab == 'tab-graph':
            from graph_utils import update_graph_store
            graph_tools_data = update_graph_store(
                add_node, remove_node, add_edge, remove_edge, generate_graph, clear_graph,
                graph_tools_data, node_id, remove_node_id, edge_source, edge_target,
                remove_edge_source, remove_edge_target, rg_graph_family, rg_num_nodes, rg_density
            )
            # Optionally sync to clustering-store if needed
            return clustering_data, graph_tools_data
        # Clustering tab: update clustering-store
        elif tab == 'tab-clustering':
            if clustering_data is None:
                clustering_data = {'clustering_method': clustering_method}
            # Fix: always provide a valid dict for .get
            graph_data = clustering_data.get('graph', graph_tools_data.get('graph', graph_tools_data))
            from graph_utils import run_clustering
            cluster_labels, y_prime = run_clustering(graph_data, clustering_method, alpha=alpha, e=e, k=k, custom_code=clustering_code, return_y=True)
            clustering_data = {
                'graph': graph_data,
                'cluster_labels': cluster_labels,
                'clustering_method': clustering_method,
                'custom_code': clustering_code,
                'alpha': alpha,
                'e': e,
                'k': k,
                'y_prime': y_prime.tolist() if hasattr(y_prime, 'tolist') else y_prime
            }
            return clustering_data, graph_tools_data
        else:
            return clustering_data, graph_tools_data

    # --- Eigen decomposition callback (Graph Tools tab only) ---
    @app.callback(
        [Output('matrix-l', 'figure'),
         Output('matrix-u', 'figure'),
         Output('matrix-s', 'figure'),
         Output('matrix-l-stack', 'figure'),
         Output('matrix-l-agg', 'figure')],
        [Input('tabs', 'value'),
         Input('graph-tools-store', 'data'),
         Input('e-input', 'value'),  # changed from e-slider to e-input
         Input('k-slider', 'value')],
        prevent_initial_call=False
    )
    def update_eigen(tab, graph_tools_data, e_input, k):
        # Ensure store is always dict, never None
        if graph_tools_data is None:
            from graph_utils import empty_graph
            graph_tools_data = empty_graph()
        if tab != 'tab-graph':
            empty = go.Figure()
            return empty, empty, empty, empty, empty
        nodes = list(graph_tools_data.get('nodes', {}).keys())
        n = len(nodes)
        if n == 0:
            empty = go.Figure()
            return empty, empty, empty, empty, empty
        e = e_input if e_input is not None else -0.5
        k = max(k, 1)
        A = np.zeros((n, n))
        for u, v in graph_tools_data.get('edges', []):
            if u in nodes and v in nodes and u != v:
                i, j = nodes.index(u), nodes.index(v)
                A[i, j] = 1; A[j, i] = 1
        degs = A.sum(axis=1)
        D_e = np.diag(degs ** e)
        L = D_e @ A @ D_e
        L = np.linalg.matrix_power(L, int(k)) if k == int(k) else L
        U, S, Vt = np.linalg.svd(L)
        figL = px.imshow(L, aspect='equal')
        figL.update_layout(title=f"L = D^{e} A^{k} D^{e}", margin=dict(l=20, r=20, t=30, b=20))
        figL.update_xaxes(title_text='Nodes', ticktext=nodes)
        figL.update_yaxes(title_text='Nodes', ticktext=nodes)
        figU = px.imshow(U, aspect='equal')
        figU.update_layout(title='U (SVD)', margin=dict(l=20, r=20, t=30, b=20))
        figU.update_yaxes(title_text='Nodes', ticktext=nodes)
        figU.update_xaxes(title_text='Eigenvectors', ticktext=[f"v{i+1}" for i in range(U.shape[1])])
        figS = go.Figure(go.Bar(x=[f"v{i+1}" for i in range(len(S))], y=S))
        figS.update_layout(title="Eigenvalues (S)", xaxis_title="Eigenvector", yaxis_title="Value", margin=dict(l=20, r=20, t=30, b=20))
        L_stack = np.expand_dims(L, axis=-1)
        opacity = 0.3
        figStack = go.Figure()
        figStack.add_trace(go.Surface(
            z=np.full_like(L, 0),
            x=np.arange(n),
            y=np.arange(n),
            surfacecolor=L,
            colorscale='Plasma',
            opacity=opacity,
            showscale=True,
            name=f"L({e},{k})"
        ))
        figStack.update_layout(
            title="Stacked L Matrix (n x n x 1)",
            scene=dict(
                xaxis_title="Node i",
                yaxis_title="Node j",
                zaxis_title="Pair Index"
            ),
            margin=dict(l=20, r=20, t=30, b=20)
        )
        L_agg = np.mean(L_stack, axis=2)
        x_title = 'Nodes'
        y_title = 'Nodes'
        x_ticktext = nodes
        y_ticktext = nodes
        aspect = 'equal'
        figAgg = px.imshow(L_agg, aspect=aspect)
        figAgg.update_layout(title=f"Aggregated L (mean, axis=2)", margin=dict(l=20, r=20, t=30, b=20))
        figAgg.update_xaxes(title_text=x_title, ticktext=x_ticktext)
        figAgg.update_yaxes(title_text=y_title, ticktext=y_ticktext)
        return figL, figU, figS, figStack, figAgg

    @app.callback(
        Output('graph-store-debug', 'children'),
        Output('clustering-store-debug', 'children'),
        Input('graph-tools-store', 'data'),
        Input('clustering-store', 'data')
    )
    def debug_store(graph_data, clustering_data):
        import json
        return json.dumps(graph_data, indent=2), json.dumps(clustering_data, indent=2)
