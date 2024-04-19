import argparse
import datetime
import logging
import os
import sys

import numpy as np
import omegaconf

import wandb

logging.basicConfig(level=logging.INFO)


sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)


from cores.provider import Provider
from cores.utils import PyGUtils
from cores.utils.plotter import MatplotlibPlotter

parser = argparse.ArgumentParser(description="Exploratory Data Analysis.")
parser.add_argument(
    "--name",
    type=str,
    required=True,
)
parser.add_argument(
    "--k_fold",
    type=str,
    required=True,
)
parser.add_argument(
    "--node_proba",
    type=float,
    default=1.0,
)
parser.add_argument(
    "--graph_proba",
    type=float,
    default=1.0,
)
parser.add_argument("--opts", required=True, nargs=argparse.REMAINDER)

args = parser.parse_args()

# Create empty cfg
cfg = omegaconf.OmegaConf.create()


args.opts.append(f"k_fold={args.k_fold}")
cfg.merge_with_dotlist(args.opts)

dataset_preparator = Provider.dataset_preparator(args.name, **cfg)

print(f"Dataset: {dataset_preparator}")

dataset_preparator.prepare()
loaders = dataset_preparator.get_dataloaders(batch_size=1)


# TODO: target_num is going to break
columns = [f"x{i}" for i in range(dataset_preparator.features_dim())]
columns += ["in_degree", "out_degree"]
columns += [f"y{i}" for i in range(dataset_preparator.target_num())]
columns += ["split"]
columns += ["graph_id"]


table_node_data = []


columns_graph = ["num_nodes", "num_edges"]

columns_graph += [f"y{i}" for i in range(dataset_preparator.target_num())]
columns_graph += ["split"]
columns_graph += ["graph_id"]

table_graph_data = []

columns_image = ["image"]
columns_image += [f"y{i}" for i in range(dataset_preparator.target_num())]
columns_image += ["split"]
columns_image += ["graph_id"]

table_image_data = []

plotter = MatplotlibPlotter()


# Use datetime to get time now in string format

now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
folder = os.path.join("results", "eda", args.name, now)

if not os.path.exists(folder):
    os.makedirs(folder)


for split_name, loader in loaders.items():
    print(f"Split: {split_name}")
    num_images = 0
    for i, batch in enumerate(loader):
        if np.random.rand() > args.graph_proba:
            continue
        y = batch.y.flatten().tolist()
        x = batch.x

        in_degree = PyGUtils.in_degree(batch.edge_index, batch.num_nodes)
        out_degree = PyGUtils.out_degree(batch.edge_index, batch.num_nodes)

        row_graph = [batch.num_nodes, batch.num_edges] + y + [split_name] + [i]
        table_graph_data.append(row_graph)

        if np.random.rand() < 0.1 and num_images < 20:
            file_path = os.path.join(folder, f"{split_name}_graph_{i}.png")
            plotter.plot_graph(graph=batch, show=False, file_path=file_path)

            image = wandb.Image(file_path)

            table_image_data.append([image] + y + [split_name] + [i])
            num_images += 1
        for j in range(x.shape[0]):
            if np.random.rand() > args.node_proba:
                continue
            x_i = x[j].tolist()
            x_i.append(in_degree[j].item())
            x_i.append(out_degree[j].item())
            row = x_i + y + [split_name] + [i]
            table_node_data.append(row)


run = wandb.init(project="cores-eda", group=args.name)


name = f"{args.name}_fold_{args.k_fold}"
raw_data_at = wandb.Artifact(name=name, type="dataset")


table_node = wandb.Table(columns=columns, data=table_node_data)
raw_data_at.add(table_node, "node_table")


table_graph = wandb.Table(columns=columns_graph, data=table_graph_data)
raw_data_at.add(table_graph, "graph_table")

table_image = wandb.Table(columns=columns_image, data=table_image_data)
raw_data_at.add(table_image, "image_table")

run.log_artifact(raw_data_at)
run.finish()
