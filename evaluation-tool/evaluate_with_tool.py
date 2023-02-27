from pathlib import Path
import os
import argparse
import sys
import subprocess
import re
import csv

regex = r'([ a-zA-Z()-]+)\t*:\s*(\d+\.?\d+)'

def run_process(exe):
    proc = subprocess.run(exe, text=True, capture_output=True)
    return proc.stdout


def main(path):
    pred = {p.stem.split('_')[0]:p for p in path.glob('*pred*.png')}
    gt = {p.stem.split('_')[0]:p for p in path.glob('*gt*.png')}
    assert len(pred) == len(gt) and all(k in gt for k in pred.keys())

    results = {}

    for id in pred.keys():
        gt_path = gt[id]
        pred_path = pred[id]
        recall_path = pred_path.parent / f'{pred_path.stem}_RWeights.dat'
        precision_path = pred_path.parent / f'{pred_path.stem}_PWeights.dat'
        if not recall_path.exists() or not precision_path.exists():
            run_process(f'{weights_exe_path} {pred_path}'.split())
        assert recall_path.exists() and precision_path.exists()
        exe = f'{metrics_exe_path.absolute()} {gt_path} {pred_path} {recall_path} {precision_path}'
        exe = exe.replace('\\', '/')
        output = run_process(exe.split())

        output = re.findall(regex, output)
        output = {k: float(v) for k, v in output}
        print(f'{id}: {output}')
        results[id] = output

    keys = list(results.values())[0].keys()
    average = {k: sum([v[k] for v in results.values()]) / len(results) for k in keys}
    results['average'] = average
    print(f'Average: {average}')

    with open(path / 'results.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames='id' + keys)
        writer.writeheader()
        for id in sorted(results.keys()):
            results[id]['id'] = id
            writer.writerow(results[id])



if __name__ == "__main__":
    weights_exe_path = Path('evaluation-tool/BinEvalWeights/BinEvalWeights.exe')
    metrics_exe_path = Path('evaluation-tool/DIBCO_metrics/DIBCO_metrics.exe')
    parser = argparse.ArgumentParser()
    parser.add_argument('path', metavar='<path>', type=str)
    args = parser.parse_args()
    path = Path(args.path)

    os.environ['PATH'] = 'C:\\Program Files\\MATLAB\\MATLAB Runtime\\v90\\runtime\\win64' + ';' + os.environ['PATH']
    main(path)
