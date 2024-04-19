from __future__ import annotations

import os

import pandas as pd
from ydata_profiling import ProfileReport

from cores.utils.eda.eda import EDA


class EDADataFrame(EDA):
    def report(
        self,
        df: pd.DataFrame,
        title: str = "DataFrame Report",
        output_file: str = "report.htaml",
        **kwargs,
    ):
        profile = ProfileReport(df, title=title, **kwargs)

        profile.to_file(output_file=os.path.join(self.root, output_file))
