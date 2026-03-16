"""
Shared utility functions for benchmark execution scripts.

These functions are used identically by evaluate_MAS_simple.py and
evaluate_MAS_orchestrated.py.
"""

import json
import logging
import os
from typing import List

from benchmark.models import BenchmarkConfig, LLMConfig

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> BenchmarkConfig:
    """Load benchmark configuration from JSON file."""
    try:
        with open(config_path, "r") as f:
            config_data = json.load(f)
        return BenchmarkConfig(**config_data)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in configuration file: {e}")
        raise


def load_llm_configs(llm_config_path: str) -> List[LLMConfig]:
    """Load LLM configuration(s) from JSON file."""
    try:
        with open(llm_config_path, "r") as f:
            config_data = json.load(f)

        # Support both single config and list of configs
        if isinstance(config_data, list):
            return [LLMConfig(**cfg) for cfg in config_data]
        else:
            return [LLMConfig(**config_data)]
    except FileNotFoundError:
        logger.error(f"LLM configuration file not found: {llm_config_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in LLM configuration file: {e}")
        raise


def skip_sample(config_file: str, output_folder: str) -> bool:
    """Check if sample has already been processed."""
    config_file_name = os.path.basename(config_file)
    output_file = os.path.join(
        output_folder, f"results_{config_file_name.replace('.json', '')}.json"
    )
    return os.path.exists(output_file)
