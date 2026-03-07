# DOMAINS_TO_RUN = ["teams", "csm", "email", "itsm", "calendar", "hr", "drive"]
# MODES_TO_RUN = ["oracle", "plus_5_tools", "plus_10_tools", "plus_15_tools"]
# MODELS = ["gpt-5-mini", "gpt-4.1-mini", "kimi-k2", "qwen3-235b-a22b-thinking", "gpt5", "qwen-4b-ngc", "qwen-30b-ngc", "qwen3-235b-a22b-instruct", "gemini_2p5", "claude"]
USE_HF_DATASET = True
HF_DATASET_REPO = "ServiceNow-AI/EnterpriseOps-Gym"

import argparse
import ray
import subprocess
import json
import os
import sys

from typing import List, Dict

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
from dataclasses import dataclass
from abc import ABC, abstractmethod
from datetime import datetime


def load_experiment_config(path: str = "conf/ray/experiment.json") -> dict:
    """Load experiment configuration from a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


@dataclass
class ResourceConfig:
    """Configuration for Ray resources."""

    llms: List[str]
    domains: List[str]
    num_llm_instances: int = 3

    def get_resource_dict(self) -> Dict[str, int]:
        """Generate Ray resource dictionary from configuration.
        Each LLM and domain has only one available resource.
        Thus only one experiment per LLM or domain can run at a time."""
        resources = {}

        # Add LLM resources
        for llm in self.llms:
            resources[f"llm_{llm}"] = self.num_llm_instances

        # Add domain resources
        for domain in self.domains:
            resources[f"domain_{domain}"] = 1

        return resources


class Experiment(ABC):
    """Abstract base class for experiments."""

    def __init__(self, llm: str, domain: str, domain_conf: dict, base_env_conf: dict, experiment_conf: dict):
        self.llm = llm
        self.domain = domain
        self.domain_conf = domain_conf
        self.base_env_conf = base_env_conf
        self.experiment_conf = experiment_conf

    @abstractmethod
    def run(self) -> str:
        """Execute the experiment logic."""
        pass

    def get_required_resources(self) -> Dict[str, int]:
        """Return the resources required for this experiment."""
        return {f"llm_{self.llm}": 1, f"domain_{self.domain}": 1}


class DefaultExperiment(Experiment):
    """Default implementation of an experiment."""

    def _get_modes(self) -> list[str]:
        # MARKER: modes can be specific to llm/domain
        return self.experiment_conf["modes"]

    def run(self) -> str:
        """
            Run experiment for a given LLM and dataset combination.

            Args:
                llm: LLM identifier (e.g., "gpt-5-mini", "gemini_2p5")
                dataset: Dataset/domain identifier (e.g., "csm", "email")

            Returns:
                Dict with experiment results and status
        """
        domain_config = self.domain_conf.get(self.domain)
        if domain_config is None:
            raise ValueError(f"Unknown domain: {self.domain!r}. Available: {list(self.domain_conf.keys())}")
        base_env_conf = self.base_env_conf
        modes = self._get_modes()

        exp = self.experiment_conf
        templates = exp["path_templates"]
        orchestrator = exp["orchestrator"]
        concurrency = exp["llm_concurrency"].get(self.llm, 5)
        num_runs = exp["num_runs"]

        print(f"[{self.llm} x {self.domain}] Starting experiment with modes: {modes}")

        results = []
        for mode in modes:
            print(f"[{self.llm} x {self.domain}] RUNNING MODE: {mode}")
            env = os.environ.copy()
            env.update(base_env_conf)
            env.update(domain_config)
            env["MODE"] = mode
            env["MODEL_NAME"] = self.llm

            timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            log_dir = os.path.join(PROJECT_ROOT, templates["log_dir"].format(orchestrator=orchestrator, llm=self.llm, domain=self.domain, mode=mode))
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, f"run_{timestamp}.log")

            output_folder = os.path.join(PROJECT_ROOT, templates["output_folder"].format(orchestrator=orchestrator, llm=self.llm, domain=self.domain, mode=mode))
            llm_config_path = os.path.join(PROJECT_ROOT, templates["llm_config"].format(llm=self.llm))

            if USE_HF_DATASET:
                data_args = [
                    "--hf_dataset", HF_DATASET_REPO,
                    "--domain", self.domain,
                    "--mode", mode,
                ]
            else:
                sample_folder = templates["sample_folder"].format(domain=self.domain, mode=mode)
                data_args = ["--configs_folder", sample_folder]

            cmd = [
                sys.executable, os.path.join(PROJECT_ROOT, "evaluate.py"),
                *data_args,
                "--llm_config", llm_config_path,
                "--orchestrator", orchestrator,
                "--output_folder", output_folder,
                "--concurrency", str(concurrency),
                "--num_runs", str(num_runs),
            ]
            if "planner_llm_config" in templates:
                cmd.extend(["--planner_llm_config", os.path.join(PROJECT_ROOT, templates["planner_llm_config"].format(llm=self.llm))])

            try:
                with open(log_file, "w") as log_f:
                    subprocess.run(
                        cmd,
                        env=env,
                        stdout=log_f,
                        stderr=subprocess.STDOUT, # Redirect stderr to stdout (and thus to log file)
                        text=True,
                        check=True,
                        timeout=3600 * 5,  # 5 hour timeout per mode
                    )

                results.append({
                    "mode": mode,
                    "status": "success",
                    "log_file": str(log_file),
                })

                print(f"[{self.llm} x {self.domain}] Completed MODE={mode} (log: {log_file})")

            except subprocess.CalledProcessError:
                results.append({
                    "mode": mode,
                    "status": "failed",
                    "log_file": str(log_file),
                    "error": "Subprocesses failed... See log_file for details.",
                })
                print(f"[{self.llm} x {self.domain}] Failed MODE={mode}: See {log_file}")

            except subprocess.TimeoutExpired:
                results.append({
                    "mode": mode,
                    "status": "timeout",
                    "log_file": str(log_file),
                    "error": "Experiment exceeded 5 hour timeout"
                })
                print(f"[{self.llm} x {self.domain}] Timeout MODE={mode}: See {log_file}")


        successful_runs = [res for res in results if res["status"] == "success"]
        return {
            "llm": self.llm,
            "domain": self.domain,
            "status": 'completed',
            "successful_runs": successful_runs,
            "total_runs": len(modes),
        }

@ray.remote
class ExperimentRunner:
    """Ray actor for running experiments."""

    def run_experiment(self, experiment: Experiment) -> str:
        """Execute a single experiment."""
        return experiment.run()


class ExperimentOrchestrator:
    """Orchestrates the execution of multiple experiments using Ray."""

    def __init__(
        self,
        config: ResourceConfig,
        domain_conf: dict,
        base_env_conf: dict,
        experiment_conf: dict,
        experiment_class=DefaultExperiment,
    ):
        self.config = config
        self.experiment_class = experiment_class
        self.domain_conf = domain_conf
        self.base_env_conf = base_env_conf
        self.experiment_conf = experiment_conf
        self.initialized = False

    def initialize(self):
        """Initialize Ray with configured resources."""
        if not self.initialized:
            ray.init(resources=self.config.get_resource_dict())
            self.initialized = True

    def create_experiments(self) -> List[Experiment]:
        """Create all experiment instances based on configuration."""
        experiments = []
        for llm in self.config.llms:
            for domain in self.config.domains:
                experiment = self.experiment_class(
                    llm, domain, self.domain_conf, self.base_env_conf, self.experiment_conf
                )
                experiments.append(experiment)
        return experiments

    def submit_experiment(self, experiment: Experiment):
        """Submit a single experiment to Ray with resource constraints."""

        # Use the functional API for simplicity
        @ray.remote
        def run_experiment_task(exp: Experiment) -> str:
            return exp.run()

        return run_experiment_task.options(
            resources=experiment.get_required_resources()
        ).remote(experiment)

    def run_all(self) -> List[str]:
        """Execute all experiments and return results."""
        self.initialize()

        experiments = self.create_experiments()
        futures = [self.submit_experiment(exp) for exp in experiments]

        # Ray automatically schedules respecting constraints
        results = ray.get(futures)
        return results

    def shutdown(self):
        """Shutdown Ray."""
        if self.initialized:
            ray.shutdown()
            self.initialized = False


# Usage example
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment_config",
        type=str,
        default="conf/ray/experiment.json",
        help="Path to experiment configuration JSON file.",
    )
    args = parser.parse_args()

    experiment_conf = load_experiment_config(args.experiment_config)

    # Pre-create output dirs in the project root
    os.makedirs(os.path.join(PROJECT_ROOT, "logs"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_ROOT, "results"), exist_ok=True)

    config = ResourceConfig(
        llms=experiment_conf["llms"],
        domains=experiment_conf["domains"],
        num_llm_instances=experiment_conf.get("num_llm_instances", 3),
    )

    with open("conf/ray/domain_conf.json", "r") as f:
        domain_conf = json.load(f)

    with open("conf/ray/base_env.json", "r") as f:
        base_env_conf = json.load(f)

    with open("conf/ray/llm_concurrency.json", "r") as f:
        experiment_conf["llm_concurrency"] = json.load(f)

    # Create orchestrator
    orchestrator = ExperimentOrchestrator(config, domain_conf, base_env_conf, experiment_conf)

    try:
        # Run all experiments
        results = orchestrator.run_all()

        # Print results
        for result in results:
            print(result)
    finally:
        # Clean up
        orchestrator.shutdown()
