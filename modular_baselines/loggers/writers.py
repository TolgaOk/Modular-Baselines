from typing import Any, Callable, Dict, List, Optional, Union, Type, Union, Protocol
import time
from collections import deque
import numpy as np
import pickle
import sys
from loguru import logger
import os
import json
import dataclasses

from modular_baselines.loggers.datalog import BaseDataLog as DataLog


class StdWriter():

    StdoutValueType: Type = Union[str, int, float]
    config: Dict[str, Any] = {
        "handlers": [
            {"sink": sys.stdout, "colorize": True,
             "format": "<level>{message}</level>"},
        ],
        "extra": {}
    }

    def __init__(self) -> None:
        logger.configure(**self.config)
        logger.level("DATA", no=38, color="<e>")
        logger.level("TITLE", no=36, color="<g>")

    def write(self, name: Optional[str], log_dump: Dict[str, Any]) -> None:
        name = name if name is not None else "- "
        logger.log("TITLE", f"- " * 8 + f"{name}" + " -" * 8)

        def _write(name: Optional[str], log_dump: Dict[str, Any], prefix: str) -> None:
            for log_name, log_value in log_dump.items():
                if isinstance(log_value, dict):
                    logger.log("TITLE", prefix + log_name)
                    _write(log_name, log_value, prefix + "   ")
                else:
                    logger.log("DATA",  f"{prefix}{log_name}: {log_value}")

        _write(None, log_dump, "")


class JsonWriter():

    def __init__(self, dirpath: str) -> None:
        self.dirpath = dirpath

    def write(self, name: Optional[str], log_dump: Dict[str, Any]) -> None:
        name = name if name is not None else "default"
        filepath = os.path.join(self.dirpath, f"{name}.json")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "a") as file_obj:
            json.dump(self._flatten_dict(log_dump), file_obj)
            file_obj.write("\n")

    def _flatten_dict(self, nested_logs: Dict[str, Any], prefix: str = None):
        items = {}
        for key, value in nested_logs.items():
            name = "/".join([prefix, key]) if prefix is not None else key
            if isinstance(value, dict):
                items = {**items, **self._flatten_dict(value, prefix=name)}
            else:
                items[name] = value
        return items


class LogConfigs():

    prefix: str = "config"

    def __init__(self,
                 config: Any,
                 dir_path: str,
                 file_name: str = "config.json"
                 ) -> None:
        super().__init__()
        self.dir = os.path.join(dir_path, self.prefix)
        self.file_name = file_name
        os.makedirs(self.dir, exist_ok=True)

        self.write(config)

    def write(self, config: Any) -> None:
        path = os.path.join(self.dir, self.file_name)
        with open(path, "w") as fobj:
            json.dump(config, fobj, default=self.serializer)

    @staticmethod
    def serializer(obj: Any):
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        return obj.jsonize()
