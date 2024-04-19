from __future__ import annotations


class EarlyStopper:
    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = "min"):
        """
        Initialize the EarlyStopper.

        Args:
            patience (int): Number of epochs with no improvement after which training will be stopped.
            min_delta (float): Minimum change in monitored metric to be considered as an improvement.
            mode (str): 'min' to minimize the metric (default) or 'max' to maximize it.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.reset_counter = 0
        self.reset_patience = 400
        self.best_metric = (
            float("inf") if mode == "min" else float("-inf")
        )  # Initialize with appropriate value
        self.early_stop = False

        self.mode = mode

    def reset(self):
        self.counter = 0
        self.reset_counter = 0
        self.best_metric = float("inf") if self.mode == "min" else float("-inf")
        self.early_stop = False

    def should_stop(self, current_metric: float) -> bool:
        """
        Check if training should stop based on the current metric.

        Args:
            current_metric (float): The value of the monitored metric at the current epoch.

        Returns:
            bool: True if training should stop, False otherwise.
        """

        self.reset_counter += 1

        if self.reset_counter >= self.reset_patience:
            self.early_stop = True
            return self.early_stop

        if (self.mode == "min" and current_metric < self.best_metric - self.min_delta) or (
            self.mode == "max" and current_metric > self.best_metric + self.min_delta
        ):
            self.best_metric = current_metric
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

        return self.early_stop
