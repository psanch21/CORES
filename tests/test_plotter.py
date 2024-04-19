import datetime
import os

import numpy as np
import pytest
import torch_geometric.datasets as pygd
import torch_geometric.datasets.graph_generator as pygdg
import torch_geometric.utils as pygu

from cores.impl.plotter import MatplotlibPlotter


@pytest.mark.parametrize("device", ["cpu"])
@pytest.mark.parametrize("mode", ["show", "save"])
def test_matplotlib(device, mode):
    plotter = MatplotlibPlotter()
    figure = plotter.create_figure()
    axis = plotter.get_axis(figure)
    if mode == "save":
        folder = os.path.join("tests", "images")
        # Create folder if it doesn't exist
        if not os.path.exists(folder):
            os.makedirs(folder)

        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        file_name = f"test_matplotlib_{timestamp}.png"

        file_path = os.path.join(folder, file_name)
        axis.plot(np.random.rand(10))
        plotter.save(figure, file_path)
        assert os.path.exists(file_path)
        # Clean up
        os.remove(file_path)

    elif mode == "show":
        axis.plot(np.random.rand(10))
        plotter.show(figure)

    plotter.close(figure)


def test_plot_graph():
    dataset = pygd.ExplainerDataset(
        graph_generator=pygdg.BAGraph(num_nodes=10, num_edges=3),
        motif_generator="house",
        num_motifs=1,
        num_graphs=2,
    )

    data = dataset[0]

    data.x = data.y.unsqueeze(1)

    plotter = MatplotlibPlotter()

    root = os.path.join("tests", "images")

    if not os.path.exists(root):
        os.makedirs(root)

    file_path = os.path.join(root, "test_plot_graph.png")

    x_unique_num = len(data.x.unique())

    color_list = MatplotlibPlotter.colorblind_colors(x_unique_num)

    graph = pygu.to_networkx(data, node_attrs=["x"], to_undirected=False)

    # Add node_color node attribute based on x node_attr
    for node, data in graph.nodes(data=True):
        x = data["x"][0]
        data["node_color"] = color_list[x]

    plotter.plot_graph(
        graph=graph, show=False, file_path=file_path, layout_fn=None, node_color_attr="node_color"
    )
