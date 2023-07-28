import sys
import csv
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np
import ast


def parse_csv(path):
    path = Path(path)
    fieldnames = []
    sweep_values = []
    with open(path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                fieldnames = row
                line_count += 1
            else:
                if len(row) != 0:
                    sweep_values.append({fieldnames[i]: row[i] for i in range(len(row))})
                    line_count += 1
        print(f'Processed {line_count} lines.')
    return sweep_values


def generate_picture(ffc, conv, metric='PSNR'):

    patch_sizes = [conv_val['path'].split('_')[-2][2:] for conv_val in conv]
    strides = [conv_val['path'].split('_')[-1][1:] for conv_val in conv]

    # ffc_vals_overlap = [float(ffc_val[metric]) for i, ffc_val in enumerate(ffc) if patch_sizes[i] != strides[i]]
    # ffc_vals_no_overlap = [float(ffc_val[metric]) for i, ffc_val in enumerate(ffc) if patch_sizes[i] == strides[i]]
    conv_vals_overlap = [float(conv_val[metric]) for i, conv_val in enumerate(conv) if patch_sizes[i] != strides[i]]
    conv_vals_no_overlap = [float(conv_val[metric]) for i, conv_val in enumerate(conv) if patch_sizes[i] == strides[i]]
    x_axis = patch_sizes[::2]

    fig, ax = plt.subplots()

    # if metric == 'PSNR':
    #     ax.set_ylim(19.5837, 20.9851)
    # elif metric == 'F-Measure':
    #     ax.set_ylim(96.4285, 98.411)
    # elif metric == 'pseudo F-Measure (Fps)':
    #     ax.set_ylim(91.3109, 93.4651)
    # elif metric == 'DRD':
    #     ax.set_ylim(2.1288, 3.1263)


    # ax.plot(x_axis, ffc_vals_overlap, color='red', linestyle='--', marker='o', label='ffc_overlap')
    # ax.plot(x_axis, ffc_vals_no_overlap, color='green', linestyle='--', marker='o',  label='ffc_no_overlap')
    ax.plot(x_axis, conv_vals_overlap, color='pink', linestyle='-', marker='o',  label='convs_overlap')
    ax.plot(x_axis, conv_vals_no_overlap, color='blue', linestyle='-', marker='o',  label='convs_no_overlap')

    ax.set_title(f'CONV {metric}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    base_save_dir = Path(r'C:\Users\fabio\Downloads\binarization_0728')
    plt.savefig(base_save_dir / f'c6a6_ranged_conv_D18_{metric}.pdf', format='pdf', dpi=1200)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--all_metrics_picture', type=str, default='false', choices=['true', 'false'])
    # parser.add_argument('--ffc_checkpoint_name', type=str)
    parser.add_argument('--conv_checkpoint_name', type=str)
    args = parser.parse_args()
    results = parse_csv(args.path)

    if args.all_metrics_picture == 'true':
        results = [ast.literal_eval(result['average']) for result in results]
        conv = [result for result in results if 'conv' in result['path']]

        # results = [result for result in results if result['id'] == 'average']
        # ffc = [result for result in results if 'FFC' in result['path']]
        # conv = [result for result in results if 'CONV' in result['path']]
        metrics = ['F-Measure', 'pseudo F-Measure (Fps)', 'PSNR', 'DRD']
    else:
        ffc = [d for d in results if d['checkpoint'] == args.ffc_checkpoint_name][0]
        conv = [d for d in results if d['checkpoint'] == args.conv_checkpoint_name][0]
        ffc_vals = [float(key_vals[1]) for i, key_vals in enumerate(ffc.items()) if i > 1]
        conv_vals = [float(key_vals[1]) for i, key_vals in enumerate(conv.items()) if i > 1]
        x = [key_vals[0].split('_')[0][2:] for i, key_vals in enumerate(ffc.items()) if i > 1]
        split = len(x) // 2
        ffc_vals_overlap = ffc_vals[:split]
        ffc_vals_no_overlap = ffc_vals[split:]
        conv_vals_overlap = conv_vals[:split]
        conv_vals_no_overlap = conv_vals[split:]
        x_axis = x[:split]
        metrics = ['PSNR']

    for metric in metrics:
        generate_picture(None, conv, metric)
    sys.exit()
