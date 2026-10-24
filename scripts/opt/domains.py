"""Per-domain paths for prompt optimization. CLI: python scripts/opt/domains.py <domain> <key>"""
import sys

DOMAINS = {
    # csm keeps its original (pre-generalization) layout so existing artifacts stay valid
    "csm": dict(bundle="final_tasks/CSM_185_Final", configs=".local_configs/csm_185_final",
                ws="scripts/opt", results="results/opt", port=8001),
    "itsm": dict(bundle="final_tasks/ITSM_181_Final", configs=".local_configs/itsm_181_final",
                 ws="scripts/opt/itsm", results="results/opt/itsm", port=8006),
    "hr": dict(bundle="final_tasks/HR_184_Final", configs=".local_configs/hr_184_final",
               ws="scripts/opt/hr", results="results/opt/hr", port=8008),
}


def get(domain):
    d = dict(DOMAINS[domain])
    d.update(domain=domain, split=f"{d['ws']}/split.json", overlays=f"{d['ws']}/overlays",
             attempts=f"{d['ws']}/attempts", logs=f"logs/opt_{domain}")
    return d


def config_path(domain, task_id):
    return f"{get(domain)['configs']}/oracle__{domain}__{task_id}.json"


if __name__ == "__main__":
    print(get(sys.argv[1])[sys.argv[2]])
