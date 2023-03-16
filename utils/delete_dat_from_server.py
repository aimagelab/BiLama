from pathlib import Path
import sys
import time


def main():
    path = Path(r'/mnt/beegfs/work/FoMo_AIISDH/scascianelli/2023_ICCV_bilama/checkpoints_data_paper')
    dat_files = list(path.rglob('*.dat'))
    num_dat_files = len(dat_files)
    for i, file in enumerate(dat_files):
        file.unlink()
        print(f'Deleted {i}/{num_dat_files}', end='\r')
    sys.exit()


if __name__ == '__main__':
    main()
