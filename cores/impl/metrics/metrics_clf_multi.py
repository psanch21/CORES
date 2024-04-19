from __future__ import annotations

import torch
import torchmetrics as tm

from cores.impl.metrics.metrics_task import TaskMetricsTorch


class MultiCLFMetricsTorch(TaskMetricsTorch):
    def __init__(self, *args, classes_num: int, **kwargs):
        self.classes_num = classes_num
        super().__init__(*args, **kwargs)

    def _add(
        self,
        full_name: str,
        average: str = "micro",
    ) -> None:
        task = "multiclass"
        num_classes = self.classes_num

        assert full_name not in self.metrics_dict

        name = self._get_name(full_name)

        if name == "accuracy":
            metric = tm.Accuracy(
                task=task,
                num_classes=num_classes,
                average=average,
                multidim_average="global",
            )
        elif name == "f1":
            metric = tm.F1Score(
                task=task,
                num_classes=num_classes,
                average=average,
                multidim_average="global",
            )
        elif name == "recall":
            metric = tm.Recall(
                task=task,
                num_classes=num_classes,
                average=average,
                multidim_average="global",
            )
        elif name == "precision":
            metric = tm.Precision(
                task=task,
                num_classes=num_classes,
                average=average,
                multidim_average="global",
            )
        else:
            raise ValueError(f"mode {name} not supported")

        self.metrics_dict[full_name] = metric.to(self.device)

    def _update(self, full_name: str, logits: float, target: int) -> None:
        probs = torch.softmax(logits, dim=-1)
        self.metrics_dict[full_name].update(probs, target)
