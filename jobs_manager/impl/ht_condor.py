from __future__ import annotations

import logging
import os

from jobs_manager.core.jobs_manager import JobsManager


class JobsManagerHTCondor(JobsManager):
    def __init__(self, output_file: str, header_file: str):

        if not output_file.endswith(".sub"):
            raise ValueError(f"The output file must be a sub file: {output_file}")

        # Check header file exists
        if not os.path.exists(header_file):
            raise FileNotFoundError(f"The header file does not exist: {header_file}")

        if not header_file.endswith(".sub"):
            raise ValueError(f"The header file must be a sub file: {header_file}")

        self.output_file = output_file
        self.output_folder = os.path.dirname(output_file)
        self.header_file = header_file

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        self.commands = []

        self.reset()

    def reset(self) -> None:
        self.commands = []

        # Create output_file with the header
        with open(self.output_file, "w") as f:
            with open(self.header_file, "r") as header_f:
                f.write(header_f.read())

    def submit(self) -> None:

        # Submit the job
        os.system(f"condor_submit {self.output_file}")

    def get_identifier(self, i: int) -> str:
        return "\$(Cluster).\$(ProcId)"

    def add_job(self, command: str, i: int) -> None:
        assert command.startswith("python ")

        # Remove the python command
        command = command.replace("python ", "")
        command_str = f'arguments = "{command}"\n'

        error_file = os.path.join(self.output_folder, "\$(Cluster).\$(ProcId).err")
        out_file = os.path.join(self.output_folder, "\$(Cluster).\$(ProcId).out")
        log_file = os.path.join(self.output_folder, "\$(Cluster).\$(ProcId).log")

        command_str += f"error = {error_file}\n"
        command_str += f"output = {out_file}\n"
        command_str += f"log = {log_file}\n"
        command_str += "queue"

        self.commands.append(command_str)
        assert len(self.commands) == i + 1

    def save(self) -> None:
        for command in self.commands:
            with open(self.output_file, "a") as f:
                f.write(f"{command}\n\n")

        logging.info(f"{len(self.commands)} jobs saved to {self.output_file}")
