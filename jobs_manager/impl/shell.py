from __future__ import annotations

import logging
import os

from jobs_manager.core.jobs_manager import JobsManager


class JobsManagerShell(JobsManager):
    def __init__(self, output_file: str):

        if not output_file.endswith(".sh"):
            raise ValueError(f"The output file must be a shell file: {self.output_file}")

        self.output_file = output_file
        self.output_folder = os.path.dirname(output_file)
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        self.commands = []

        self.reset()

    def reset(self) -> None:
        self.commands = []

        # Initialize the output file
        with open(self.output_file, "w") as f:
            f.write("#!/bin/bash\n")

    def submit(self) -> None:

        # Submit the job
        os.system(f"bash {self.output_file}")

    def get_identifier(self, i: int) -> str:
        return f"job_{i}"

    def add_job(self, command: str, i: int) -> None:

        self.commands.append(command)
        assert len(self.commands) == i + 1

    def save(self) -> None:
        for command in self.commands:
            with open(self.output_file, "a") as f:
                f.write(f"{command}\n\n")

        logging.info(f"Jobs saved to {self.output_file}")
