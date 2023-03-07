import sys
import csv
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import numpy as np


def main(path):
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
                sweep_values.append({fieldnames[i]: row[i] for i in range(len(row))})
                line_count += 1
        print(f'Processed {line_count} lines.')

    # Find the dictionary in sweep_values where 'checkpoint' has the value 'best'
    ffc = [d for d in sweep_values if
           d['checkpoint'] == 'FFC_3RB_catSKIP_2UL_3DS_charLOSS_cosiSCHE_3d22_best_psnr_test.pth'][0]
    conv = [d for d in sweep_values if
           d['checkpoint'] == 'CONV_3RB_catSKIP_2UL_3DS_charLOSS_cosiSCHE_651e_best_psnr_test.pth'][0]

    ffc_vals = [float(key_vals[1]) for i, key_vals in enumerate(ffc.items()) if i > 1]
    conv_vals = [float(key_vals[1]) for i, key_vals in enumerate(conv.items()) if i > 1]
    x = [key_vals[0].split('_')[0][2:] for i, key_vals in enumerate(ffc.items()) if i > 1]
    split = len(x) // 2
    ffc_vals_overlap = ffc_vals[:split]
    ffc_vals_no_overlap = ffc_vals[split:]
    conv_vals_overlap = conv_vals[:split]
    conv_vals_no_overlap = conv_vals[split:]
    x_axis = x[:split]

    fig, ax = plt.subplots()
    ax.plot(x_axis, ffc_vals_overlap, color='red', linestyle='--', label='ffc_overlap')
    ax.plot(x_axis, ffc_vals_no_overlap, color='green', linestyle='--', label='ffc_no_overlap')
    ax.plot(x_axis, conv_vals_overlap, color='pink', linestyle='-', label='convs_overlap')
    ax.plot(x_axis, conv_vals_no_overlap, color='blue', linestyle='-', label='convs_no_overlap')

    ax.set_title('FFC vs CONV')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    plt.savefig('ffc_vs_conv.pdf')
    plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    args = parser.parse_args()
    main(args.path)
    sys.exit()
