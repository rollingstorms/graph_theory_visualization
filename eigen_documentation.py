from dash import html

DOCS_CONTENT = [
    html.H1("Graph Theory Dashboard Documentation"),

    html.H2("Overview"),
    html.P("This interactive dashboard enables users to build, visualize, and analyze graphs using advanced tools from spectral graph theory. It leverages Dash and Plotly to provide intuitive, interactive visualizations and analysis capabilities."),

    html.H2("Main Features"),

    html.H3("Random Graph Generator"),
    html.P("Generate random graphs from several common families:"),
    html.Ul([
        html.Li(html.A("Erdős–Rényi (ER)", href="https://en.wikipedia.org/wiki/Erd%C5%91s%E2%80%93R%C3%A9nyi_model", target="_blank")),
        html.Li(html.A("Barabási–Albert (BA)", href="https://en.wikipedia.org/wiki/Barab%C3%A1si%E2%80%93Albert_model", target="_blank")),
        html.Li(html.A("Watts–Strogatz (WS)", href="https://en.wikipedia.org/wiki/Watts%E2%80%93Strogatz_model", target="_blank")),
        html.Li(html.A("Scale-Free (SF)", href="https://en.wikipedia.org/wiki/Scale-free_network", target="_blank")),
        html.Li(html.A("Star Graph", href="https://en.wikipedia.org/wiki/Star_(graph_theory)", target="_blank")),
        html.Li(html.A("Lattice Graph", href="https://en.wikipedia.org/wiki/Lattice_graph", target="_blank")),
        html.Li(html.A("Delaunay Triangulation", href="https://en.wikipedia.org/wiki/Delaunay_triangulation", target="_blank")),
    ]),

    html.H3("Matrix Decomposition Tools"),
    html.P([
        "The dashboard uses spectral decomposition methods on the graph Laplacian matrix. You can modify parameters ",
        html.Code("E"), " and ", html.Code("K"), " to explore different spectral filters defined by the operator:"
    ]),
    html.Pre("L = (D^E A D^E)^K", style={"background": "#f5f5f5", "padding": "10px", "borderRadius": "5px"}),
    html.P([
        html.Strong("Laplacian Matrix"), ": Encodes graph connectivity and structure. See ",
        html.A("Graph Laplacian", href="https://en.wikipedia.org/wiki/Laplacian_matrix", target="_blank"),
        " for detailed explanations."
    ]),

    html.H3("Spectral Graph Theory"),
    html.P("Spectral graph theory involves analyzing properties of graphs using eigenvectors and eigenvalues of matrices like the Laplacian."),
    html.Ul([
        html.Li(html.A("Spectral Graph Theory Overview", href="https://en.wikipedia.org/wiki/Spectral_graph_theory", target="_blank")),
        html.Li(html.A("Eigenvalues and Eigenvectors", href="https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors", target="_blank")),
        html.Li(html.A("Singular Value Decomposition (SVD)", href="https://en.wikipedia.org/wiki/Singular_value_decomposition", target="_blank"))
    ]),

    html.H3("Interactive Visualizations"),
    html.Ul([
        html.Li(html.Strong("Graph Visualization:"), " Nodes and edges colored by Laplacian values. Hover for details."),
        html.Li(html.Strong("Matrix L:"), " Laplacian matrix visualization."),
        html.Li(html.Strong("Matrix U:"), " Eigenvectors from the SVD of L."),
        html.Li(html.Strong("Matrix S:"), " Singular values (spectrum)."),
        html.Li(html.Strong("Matrix Stack/Aggregation:"), " Compare and aggregate multiple parameter (E, K) combinations.")
    ]),

    html.H3("Graph Properties Table"),
    html.P("Displays fundamental properties of the graph including:"),
    html.Ul([
        html.Li(html.Strong("Number of Nodes and Edges:"), " Basic size measures."),
        html.Li(html.Strong("Connected Components:"), " Number of isolated subgraphs."),
        html.Li(html.Strong("Diameter:"), " Longest shortest path (only for connected graphs)."),
        html.Li(html.Strong("Average Degree:"), " Average number of connections per node."),
        html.Li(html.Strong("Average Clustering Coefficient:"), " Indicates clustering tendency."),
        html.Li(html.Strong("Density:"), " Proportion of potential connections realized.")
    ]),

    html.H3("Using the Dashboard"),
    html.P("Steps to get started:"),
    html.Ol([
        html.Li("Select a graph family and generate a graph."),
        html.Li("Adjust spectral decomposition parameters (E and K) to explore spectral properties."),
        html.Li("Add, remove, or edit nodes and edges to see how graph properties and spectral decompositions change."),
        html.Li("Use aggregation and parameter lists for advanced comparisons.")
    ]),

    html.H3("Tips and Advanced Usage"),
    html.Ul([
        html.Li("Hovering over nodes and edges provides specific Laplacian values and connectivity information."),
        html.Li("Try varying E between -0.5 and 0.5 to explore different normalization effects."),
        html.Li("Experiment with higher K values to understand graph diffusion dynamics."),
        html.Li("Regularly refer to this documentation for clarification and deeper insights.")
    ]),

    html.H3("Further Reading and Resources"),
    html.Ul([
        html.Li(html.A("NetworkX Library Documentation", href="https://networkx.org/documentation/stable/", target="_blank")),
        html.Li(html.A("Dash Documentation", href="https://dash.plotly.com/", target="_blank")),
        html.Li(html.A("Plotly Graph Objects Reference", href="https://plotly.com/python/graph-objects/", target="_blank")),
        html.Li(html.A("Scipy Spatial Documentation (Delaunay)", href="https://docs.scipy.org/doc/scipy/reference/spatial.html", target="_blank"))
    ])
]