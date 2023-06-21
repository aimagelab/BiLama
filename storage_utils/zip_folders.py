from pathlib import Path
import shutil
import sys
import json

def main():

    paths = [
        Path(r'/mnt/beegfs/work/FoMo_AIISDH/scascianelli/2023_ICCV_bilama/data/PHIBD'),
        Path(r'/mnt/beegfs/work/FoMo_AIISDH/scascianelli/2023_ICCV_bilama/checkpoints_data_paper'),
        Path(r'/mnt/beegfs/work/FoMo_AIISDH/scascianelli/2023_ICCV_bilama/data/BiLama_binarization_results_20230314'),
        Path(r'/mnt/beegfs/work/FoMo_AIISDH/scascianelli/2023_ICCV_bilama/data/BiLama_binarization_results_20230315'),
        Path(
            r'/mnt/beegfs/work/FoMo_AIISDH/scascianelli/2023_ICCV_bilama/data/BiLama_binarization_results_20230315_PHIBD'),
        Path(r'/mnt/beegfs/work/FoMo_AIISDH/scascianelli/2023_ICCV_bilama/data/BiLama_binarization_results_20230316'),
        Path(r'/mnt/beegfs/work/FoMo_AIISDH/scascianelli/2023_ICCV_bilama/data/BiLama_binarization_results_20230507'),
        Path(r'/mnt/beegfs/work/FoMo_AIISDH/scascianelli/2023_ICCV_bilama/data/BiLama_binarization_results_20230508'),
        Path(r'/mnt/beegfs/work/FoMo_AIISDH/scascianelli/2023_ICCV_bilama/data/BiLama_binarization_results_fquattrini'),
        Path(
            r'/mnt/beegfs/work/FoMo_AIISDH/scascianelli/2023_ICCV_bilama/data/D19_BiLama_binarization_results_20230508'),
        Path(
            r'/mnt/beegfs/work/FoMo_AIISDH/scascianelli/2023_ICCV_bilama/data/D19BiLama_binarization_results_20230508'),
        Path(r'/mnt/beegfs/work/FoMo_AIISDH/scascianelli/2023_ICCV_bilama/data/DEGAN_results'),
        Path(r'/mnt/beegfs/work/FoMo_AIISDH/scascianelli/2023_ICCV_bilama/data/DocEnTr_results'),
        Path(r'/mnt/beegfs/work/FoMo_AIISDH/scascianelli/2023_ICCV_bilama/data/FourBi_results'),
        Path(r'/mnt/beegfs/work/FoMo_AIISDH/scascianelli/2023_ICCV_bilama/data/FourBi_results_3RB')
    ]

    # paths = [
    #     Path(r'/mnt/beegfs/work/FoMo_AIISDH/scascianelli/2023_ICCV_bilama/checkpoints/12RB'),
    #     Path(r'/mnt/beegfs/work/FoMo_AIISDH/scascianelli/2023_ICCV_bilama/checkpoints/15RB'),
    #     Path(r'/mnt/beegfs/work/FoMo_AIISDH/scascianelli/2023_ICCV_bilama/checkpoints/2RB'),
    #     Path(r'/mnt/beegfs/work/FoMo_AIISDH/scascianelli/2023_ICCV_bilama/checkpoints/3RB'),
    #     Path(r'/mnt/beegfs/work/FoMo_AIISDH/scascianelli/2023_ICCV_bilama/checkpoints/6RB'),
    #     Path(r'/mnt/beegfs/work/FoMo_AIISDH/scascianelli/2023_ICCV_bilama/checkpoints/9RB')
    #     ]

    for i, path in enumerate(paths):
        print(f'Processing {i + 1}/{len(paths)}', end='\r')
        if Path(f'{path}.zip').is_file():
            continue
        shutil.make_archive(f'{path}', 'zip', path)

    print('Done!')


if __name__ == '__main__':
    main()
    sys.exit()

