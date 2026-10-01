"""DataSea human-demonstration collection layer for EnterpriseOps-Gym.

Everything in this package sits on top of the upstream benchmark code
(`benchmark/`, `evaluate.py`) and reuses its MCP client, database seeding,
and verifier machinery without modifying it.
"""

import os

DATASEA_VERSION = "0.1.0"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASEA_DIR = os.path.dirname(os.path.abspath(__file__))
RUNTIME_DIR = os.environ.get("DATASEA_RUNTIME_DIR", os.path.join(DATASEA_DIR, "runs"))
