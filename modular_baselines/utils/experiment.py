from typing import Any
import os
from sacred import Experiment


def record_log_files(ex: Experiment, log_dir: str) -> None:
    for file_name in os.listdir(log_dir):
        ex.add_artifact(os.path.join(log_dir, file_name))
