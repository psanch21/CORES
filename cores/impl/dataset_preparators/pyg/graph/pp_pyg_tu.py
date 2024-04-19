from __future__ import annotations

from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import Compose

from cores.impl.dataset_preparators.pyg.graph.pp_pyg_graph import PyGGraphPreparator


class TUPreparator(PyGGraphPreparator):
    @staticmethod
    def random_kwargs(seed: int):
        kwargs = PyGGraphPreparator.random_kwargs(seed)

        return kwargs

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def get_transform_fn(self):
        def transform(data):
            data.y = data.y.unsqueeze(-1)
            return data

        return transform

    def cleaned(self):
        return True

    def _get_dataset_raw(self) -> TUDataset:
        def pp_fn(data):
            # if self.pp_x is not None:
            #     dtype = data.x.dtype
            #     device = data.x.device
            #     x_np = self.pp_x.transform(data.x.numpy())

            #     data.x = torch.from_numpy(x_np).to(dtype=dtype)

            return data

        transform = Compose([pp_fn, self.get_transform_fn()])

        if self.is_dense:
            dataset = TUDataset(
                root=self.root,
                name=self.name.upper(),
                transform=transform,
                use_node_attr=True,
                cleaned=self.cleaned(),
            )
        else:
            dataset = TUDataset(
                root=self.root,
                name=self.name.upper(),
                transform=transform,
                pre_transform=None,
                pre_filter=None,
                use_node_attr=True,
                use_edge_attr=True,
                cleaned=self.cleaned(),
            )
        return dataset

    def classes_num(self) -> int:
        raise NotImplementedError

    def edge_attr_dim(self) -> int:
        raise NotImplementedError

    def features_dim(self) -> int:
        raise NotImplementedError

    def target_dim(self) -> int:
        raise NotImplementedError
