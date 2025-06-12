# Graph Theory Dashboard

An interactive Dash-based tool for exploring and analyzing graphs through spectral graph theory, Laplacian decomposition, and interactive visualizations.

## Features

- **Interactive Graph Generation**:
  - Erdős–Rényi, Barabási–Albert, Watts–Strogatz, Scale-Free, Star, Lattice, and Delaunay Triangulation.

- **Spectral Analysis**:
  - Compute and visualize the Laplacian matrix and its eigen decomposition.
  - Explore spectral filters defined by:
    $$L = (D^E A D^E)^K$$

- **Graph Manipulation**:
  - Add, remove, and edit nodes and edges interactively.

- **Matrix Aggregation**:
  - Analyze effects of multiple (E, K) parameter pairs.
- **Custom Clustering**:
  - Select built-in algorithms or supply Python code for your own clustering routine.

- **Detailed In-App Documentation**:
  - Comprehensive explanations of spectral graph theory concepts.

## Installation

Clone the repository and install required dependencies:

```bash
git clone <repo-url>
cd <repo-directory>
pip install -r requirements.txt
```

## Running the Dashboard

Launch the dashboard locally:

```bash
python eigen_app.py
```

Open your browser and navigate to:

```
http://127.0.0.1:8050
```

## Project Structure

```
.
├── eigen_app.py            # Main Dash application
├── eigen_documentation.py  # Documentation content
├── requirements.txt        # Python dependencies
├── readme.md               # Readme
```

## Planned Features

- Clustering visualizations and community detection
- Graph import/export functionality
- Enhanced node and edge manipulation tools

## Resources

- [Dash documentation](https://dash.plotly.com/)
- [NetworkX documentation](https://networkx.org/documentation/stable/)
- [Plotly Python reference](https://plotly.com/python/)

## Contributions

Contributions, bug reports, and feature requests are welcome. Please open an issue or submit a pull request.

## License

MIT License.
