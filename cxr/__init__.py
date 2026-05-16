"""Bootstrap: warning filters, env loading, and torch thread configuration.

This module is imported before any submodule, so it runs before transformers
is imported anywhere in the package. The warning filters must be installed
here to suppress the noisy transformers deprecation messages.
"""
from __future__ import annotations

import logging
import os
import warnings

warnings.filterwarnings("ignore", message=r".*Accessing `__path__` from.*")


class _DropPathDeprecation(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "Accessing `__path__` from" not in record.getMessage()


logging.getLogger("transformers").addFilter(_DropPathDeprecation())

from dotenv import load_dotenv

load_dotenv()

import torch

torch.set_num_threads(os.cpu_count() or 1)
