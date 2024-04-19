import argparse
import logging
import os
import sys
from datetime import datetime

logging.basicConfig(level=logging.INFO)


sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
)
from jobs_manager import JobsManagerShell
from sweep import RandomSweep

# Argsparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create sweeps")
    parser.add_argument(
        "--main_file",
        type=str,
        default="cores/delivery/cli/cores_train.py",
        help="Path to the main file",
    )

    parser.add_argument(
        "--sweep_file",
        type=str,
        default="config/sweep/gnn.yaml",
        help="Path to the sweep file",
    )

    parser.add_argument("--count", default=100, type=int, help="Number of runs")
    parser.add_argument("--start", default=0, type=int, help="Number of runs")
    parser.add_argument("--seed", default=42, type=int, help="Seed")
    parser.add_argument("--k_folds", default="1+2+3+4+5", type=str, help="K-folds")
    parser.add_argument("--root_folder", type=str, default="results/test")
    parser.add_argument("--jobs_manager", type=str, default="")

    parser.add_argument("--opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # Get current time in human readable format

    now = datetime.now()
    now_hf = now.strftime("%Y%m%d_%H%M%S")
    args.root_folder = os.path.join(args.root_folder, now_hf)

    param_files = []
    extra_opts = ""

    if args.opts is not None:
        param_files = []
        config_files = []
        for opt in args.opts:
            opt_name, opt_value = opt.split("=")
            if opt_name.startswith("param_"):
                param_files.append(opt_value)
            elif opt_name.startswith("config_"):
                config_files.append(f"--{opt_name} {opt_value}")
            else:
                extra_opts += f" {opt}"

        config_files = " ".join(config_files)
    else:
        param_files = None
        config_files = ""

    sweep = RandomSweep(sweep_file_path=args.sweep_file, param_files=param_files)

    configs = sweep.sample(seed=args.seed, count=args.count, start=args.start)
    commands = ""

    jobs_manager = JobsManagerShell.from_dotlist(args.jobs_manager)
    j = 0
    for i, config in enumerate(configs):
        for k_fold in args.k_folds.split("+"):
            id_str = jobs_manager.get_identifier(j)
            root_folder_j = os.path.join(args.root_folder, id_str)
            command_str = f"python {args.main_file} {config_files} --opts {config}{extra_opts} data_kwargs.k_fold={k_fold} root_folder={root_folder_j}"

            jobs_manager.add_job(command_str, j)
            j += 1

    jobs_manager.save()
