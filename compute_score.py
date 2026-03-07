import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from tabulate import tabulate

true = True
false = False
null = None

def get_score(sample):
    return (sample['statistics']['overall_success_rate'], sample['statistics']['verifier_level_pass_rate'])

def process_mode(results_folder, mode):
    file_list = []
    for root, _, files in os.walk(results_folder):
        for file in files:
            if file.endswith('.json'):
                file_list.append(os.path.relpath(os.path.join(root, file), results_folder))

    overall_success_rates = []
    verifier_level_pass_rates = []
    errors = []

    for idx, file_name in enumerate(tqdm(file_list, desc=f"Processing {mode}")):
        if not file_name.endswith('.json'):
            continue
        file_path = os.path.join(results_folder, file_name)
        with open(file_path, 'r') as f:
            # if '+5' not in file_path.lower():
            #     continue
            file_content = f.read()
            result_data = json.loads(file_content)
            has_error = any(run.get("error") for run in result_data.get("runs", []))
        overall_success_rate, verifier_level_pass_rate = get_score(result_data)
        overall_success_rates.append(overall_success_rate)
        verifier_level_pass_rates.append(verifier_level_pass_rate)
        errors.append(1 if has_error else 0)

    avg_overall_success_rate = sum(overall_success_rates) / len(overall_success_rates) if overall_success_rates else 0
    avg_verifier_level_pass_rate = sum(verifier_level_pass_rates) / len(verifier_level_pass_rates) if verifier_level_pass_rates else 0

    return {
        'mode': mode,
        'total_files': len(overall_success_rates),
        'files_with_errors': sum(errors),
        'avg_overall_success_rate': avg_overall_success_rate * 100.0,
        'avg_verifier_pass_rate': avg_verifier_level_pass_rate * 100.0
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_folder', type=str, required=True, help='Path to the input JSON file')

    args = parser.parse_args()
    results_folder = args.results_folder

    # Collect results from all modes
    all_results = []

    # For each mode (folder) in results_folder
    for mode in sorted(os.listdir(results_folder)):
        mode_folder = os.path.join(results_folder, mode)
        if os.path.isdir(mode_folder):
            result = process_mode(mode_folder, mode)
            all_results.append(result)

    # Print results as a nice table
    if all_results:
        print("\n" + "="*100)
        print("FINAL RESULTS")
        print("="*100 + "\n")

        headers = ["Mode", "Total Files", "Files w/ Errors", "Avg Success Rate (%)", "Avg Verifier Pass (%)"]
        table_data = []

        for result in all_results:
            row = [
                result['mode'],
                result['total_files'],
                f"\033[91m{result['files_with_errors']}\033[0m" if result['files_with_errors'] > 0 else "0",
                f"{result['avg_overall_success_rate']:.2f}",
                f"{result['avg_verifier_pass_rate']:.2f}"
            ]
            table_data.append(row)

        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        print()


if __name__ == "__main__":
    main()