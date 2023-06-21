from pathlib import Path
import shutil
import sys
import json

def main():
    path = Path(r'/mnt/beegfs/work/FoMo_AIISDH/scascianelli/2023_ICCV_bilama/checkpoints')
    rb_layers_nums = [3, 6, 9, 12, 15]
    checkpoints = {rb_layer_num: sorted(list(path.glob(f'*{rb_layer_num}RB*'))) for rb_layer_num in rb_layers_nums}

    for rb_layer_num in rb_layers_nums:
        rb_layer_num_path = path / f'{rb_layer_num}RB'
        rb_layer_num_path.mkdir(exist_ok=True)
        num_checkpoints = len([c for c in checkpoints[rb_layer_num] if c.is_file()])
        print(f'Found {num_checkpoints} checkpoints for {rb_layer_num}RB layer(s)')
        for i, checkpoint in enumerate(checkpoints[rb_layer_num]):
            if checkpoint.is_file():
                checkpoint.rename(rb_layer_num_path / checkpoint.name)
                print(f'Moving {i + 1}/{num_checkpoints}', end='\r')
        print('\n')

    print('\nDone!')


if __name__ == '__main__':
    main()
    sys.exit()