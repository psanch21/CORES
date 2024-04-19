from __future__ import annotations

import copy
from random import sample
from typing import Dict, List, Optional, Tuple

import torch
import torch_geometric.data as pygd
import torch_geometric.utils as pygu

import cores.core.values.constants as cte
from cores.utils.torch import TorchUtils


class PyGUtils:
    @staticmethod
    def in_degree(edge_index: torch.LongTensor, num_nodes: int) -> torch.LongTensor:
        dst_nodes = edge_index[1]
        in_degree = pygu.degree(index=dst_nodes, num_nodes=num_nodes)
        return in_degree

    @staticmethod
    def out_degree(edge_index: torch.LongTensor, num_nodes: int) -> torch.LongTensor:
        src_nodes = edge_index[0]
        out_degree = pygu.degree(index=src_nodes, num_nodes=num_nodes)
        return out_degree

    @staticmethod
    def transductive_split_data(
        data, split_sizes: List[float], task_domain: cte.TaskDomains, k_fold: int
    ):
        if task_domain == cte.TaskDomains.NODE:
            idx_splits = TorchUtils.split_into_segments(
                original_len=data.node_feature.shape[0],
                segment_sizes=split_sizes,
                k_fold=k_fold,
            )

            datasets = []
            for nodes_split_i in idx_splits:
                data_i = copy.deepcopy(data)
                setattr(data_i, "node_label_index", nodes_split_i)
                datasets.append(data_i)
        else:
            raise NotImplementedError(f"Task domain {task_domain} not implemented")

        return datasets

    @staticmethod
    def remove_edges_from_batch(batch: pygd.Batch, edges_to_remove: List[int]):
        if len(edges_to_remove) == 0:
            return batch
        edge_index, edge_attr = PyGUtils.remove_edges(
            edges_to_remove=edges_to_remove,
            num_edges=batch.edge_index.shape[1],
            edge_index=batch.edge_index,
            edge_attr=batch.edge_attr,
        )

        batch.edge_index = edge_index
        batch.edge_attr = edge_attr
        return batch

    @staticmethod
    def remove_edges(
        edges_to_remove: List[int],
        num_edges: int,
        edge_index: torch.LongTensor,
        edge_attr: Optional[torch.FloatTensor] = None,
    ):
        assert isinstance(edges_to_remove, list)
        assert isinstance(num_edges, int)

        edges_to_keep = torch.tensor(
            list(set(range(num_edges)) - set(edges_to_remove)), dtype=torch.long
        )
        edge_index = edge_index[:, edges_to_keep]

        if edge_attr is not None:
            edge_attr = edge_attr[edges_to_keep, :]
        return edge_index, edge_attr

    @staticmethod
    def remove_nodes_from_batch(
        batch: pygd.Batch,
        nodes_idx: List[int] = None,
        mode="remove",
        relabel_nodes: bool = False,
        has_batch_att: bool = True,
    ):
        if len(nodes_idx) == 0:
            return batch
        n_mask, edge_index, edge_attr = PyGUtils.remove_nodes(
            num_nodes=batch.x.shape[0],
            edge_index=batch.edge_index,
            nodes_idx=nodes_idx,
            mode=mode,
            relabel_nodes=relabel_nodes,
            edge_attr=batch.edge_attr,
        )
        batch.edge_index = edge_index
        batch.edge_attr = edge_attr
        batch.x = batch.x[n_mask]
        if has_batch_att:
            batch.batch = batch.batch[n_mask]
            # batch.node_id = batch.node_id[n_mask]
        return batch

    @staticmethod
    def remove_nodes_from_batch_2(
        batch: pygd.Batch,
        nodes_idx: List[int] = None,
        mode="remove",
        relabel_nodes: bool = False,
        has_batch_att: bool = True,
    ):
        # remove nodes from batch using data_list
        if len(nodes_idx) == 0:
            return batch

        data_list = batch.to_data_list()
        data_list_out = []

        node_idx_tensor = torch.tensor(nodes_idx)
        for data_i in data_list:
            cond = node_idx_tensor < data_i.num_nodes
            nodes_idx_i = node_idx_tensor[cond]

            node_idx_tensor = node_idx_tensor[~cond]
            node_idx_tensor -= data_i.num_nodes

            n_mask, edge_index, edge_attr = PyGUtils.remove_nodes(
                num_nodes=data_i.x.shape[0],
                edge_index=data_i.edge_index,
                nodes_idx=nodes_idx_i.tolist(),
                mode=mode,
                relabel_nodes=relabel_nodes,
                edge_attr=data_i.edge_attr,
            )
            data_i.edge_index = edge_index
            data_i.edge_attr = edge_attr
            data_i.x = data_i.x[n_mask]
            data_list_out.append(data_i)

        batch = pygd.Batch.from_data_list(data_list_out)

        batch.batch = batch.batch.to(batch.x.device)
        return batch

    @staticmethod
    def remove_nodes(
        num_nodes: int,
        edge_index: torch.LongTensor,
        nodes_idx: List[int] = None,
        mode="remove",
        relabel_nodes=False,
        edge_attr=None,
    ):
        assert isinstance(nodes_idx, list)
        assert isinstance(num_nodes, int)

        device = edge_index.device
        if mode == "remove":
            nodes_to_keep = torch.tensor(
                list(set(range(num_nodes)) - set(nodes_idx)), dtype=torch.long
            )
        elif mode == "keep":
            nodes_to_keep = torch.tensor(nodes_idx, dtype=torch.long)
        else:
            raise NotImplementedError

        n_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
        n_mask[nodes_to_keep] = 1

        if relabel_nodes:
            n_idx = torch.zeros(num_nodes, dtype=torch.long, device=device)
            n_idx[nodes_to_keep] = torch.arange(nodes_to_keep.size(0), device=device)

        mask = n_mask[edge_index[0]] & n_mask[edge_index[1]]

        edge_index = edge_index[:, mask]
        edge_attr = edge_attr[mask] if edge_attr is not None else None

        if relabel_nodes:
            edge_index = n_idx[edge_index]

        return n_mask, edge_index, edge_attr

    @staticmethod
    def add_x_noise(batch: pygd.Batch, eps: float, inplace: bool = True) -> pygd.Batch:
        if inplace:
            batch.x = batch.x + torch.randn_like(batch.x) * eps
            return batch
        else:
            batch = batch.copy()
            batch.x = batch.x + torch.randn_like(batch.x) * eps
            return batch

    @staticmethod
    def add_edge_noise_batch(
        batch: pygd.Batch, p: float, sort: bool = True, inplace: bool = True
    ) -> pygd.Batch:
        if not inplace:
            batch = batch.copy()

        data_list = batch.to_data_list()

        data_list_out = []

        for data in data_list:
            edge_index, edge_attr = PyGUtils.add_edge_noise(
                edge_index=data.edge_index,
                num_nodes=data.x.size(0),
                p=p,
                edge_attr=data.edge_attr,
                sort=sort,
            )
            data.edge_index = edge_index
            data.edge_attr = edge_attr
            data_list_out.append(data)

        return pygd.Batch.from_data_list(data_list_out)

    @staticmethod
    def add_edge_noise(
        edge_index: torch.LongTensor, num_nodes: int, p: float, edge_attr=None, sort=True
    ) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        is_undirected = pygu.is_undirected(edge_index)
        assert p >= 0.0
        if p == 0.0:
            return edge_index, edge_attr
        edge_set = set(map(tuple, edge_index.transpose(0, 1).tolist()))
        num_of_new_edge = int((edge_index.size(1) // 2) * p)
        if num_of_new_edge == 0:
            return edge_index, edge_attr
        to_add = list()

        max_num_edges = num_nodes**2

        num_edge_samples = min(num_of_new_edge + len(edge_set) + num_nodes, max_num_edges)

        new_edges = sample(range(1, num_nodes**2 + 1), num_edge_samples)
        c = 0
        for i in new_edges:
            if c >= num_of_new_edge:
                break
            s = ((i - 1) // num_nodes) + 1
            t = i - (s - 1) * num_nodes
            s -= 1
            t -= 1
            if s != t and (s, t) not in edge_set:
                c += 1
                to_add.append([s, t])
                to_add.append([t, s])
                edge_set.add((s, t))
                edge_set.add((t, s))

        if len(to_add) == 0:
            return edge_index, edge_attr

        new_edge_index = torch.cat(
            [edge_index.to("cpu"), torch.LongTensor(to_add).transpose(0, 1)], dim=1
        )

        if edge_attr is not None:
            new_edge_attr = torch.cat(
                [edge_attr, torch.zeros(len(to_add), edge_attr.size(1)).to(edge_attr.device)], dim=0
            )
        else:
            new_edge_attr = None

        if sort:
            if edge_attr is not None:
                new_edge_index, new_edge_attr = pygu.sort_edge_index(
                    edge_index=new_edge_index, edge_attr=new_edge_attr, num_nodes=num_nodes
                )
            else:
                new_edge_index = pygu.sort_edge_index(
                    edge_index=new_edge_index, num_nodes=num_nodes
                )

        if is_undirected:
            if edge_attr is not None:
                new_edge_index, new_edge_attr = pygu.to_undirected(
                    edge_index=new_edge_index, edge_attr=new_edge_attr, num_nodes=num_nodes
                )
            else:
                new_edge_index = pygu.to_undirected(edge_index=new_edge_index, num_nodes=num_nodes)

        return new_edge_index, new_edge_attr

    @staticmethod
    def compute_stats_batch(batch: pygd.Batch, num_samples: int = 1) -> Dict[str, torch.Tensor]:
        data_list = batch.to_data_list()
        stats = {}
        for graph_i in data_list:
            stats_i = PyGUtils.compute_stats_graph(graph_i)
            for name, value in stats_i.items():
                if name not in stats:
                    stats[name] = []
                stats[name].append(value)

        output = {}
        for name, values in stats.items():
            my_tensor = torch.tensor(values)
            if num_samples > 1:
                my_tensor = my_tensor.view(-1, num_samples).float().mean(1)

            output[name] = my_tensor

        return output

    @staticmethod
    def compute_stats_graph(graph: pygd.Data):
        stats = {}
        stats["num_nodes"] = graph.x.shape[0]
        stats["num_edges"] = graph.edge_index.shape[1]

        # label = graph.y.item()
        # stats[f"num_nodes_{label}"] = graph.x.shape[0]
        # stats[f"num_edges_{label}"] = graph.edge_index.shape[1]

        return stats
