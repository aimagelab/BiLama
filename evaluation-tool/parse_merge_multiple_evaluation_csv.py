from pathlib import Path
import os
import argparse
import sys
import subprocess
import re
import csv


def main(path):
    print(f'Processing {path}')
    csv_path = path / 'results.csv'

    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        results = [{k: v for k, v in row.items()} for row in reader]

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_save_path', type=str, required=True)
    parser.add_argument('--paths', type=str, nargs='+', required=True)
    args = parser.parse_args()

    results_all = []
    for path in args.paths:
        print(f'Processing {path}')
        results_all.extend(main(Path(path)))

    csv_save_path = Path(args.csv_save_path)
    with open(csv_save_path / 'results_all.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(results_all[0].keys()))
        writer.writeheader()
        for result in results_all:
            writer.writerow(result)
