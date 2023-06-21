from pathlib import Path
import shutil
import time
import json

def main():
    bevagna_paths = [
        Path(r'/home/shared/bilama_bevagna/csv_results_bevagna'),
        Path(r'/home/shared/bilama_bevagna/images_bevagna'),
        Path(r'/home/shared/bilama_bevagna/checkpoints_bevagna')
    ]

    server_paths = [
        Path(r'/mnt/2023_ICCV_bilama/checkpoints_data_paper/csv_results/bevagna'),
        Path(r'/mnt/2023_ICCV_bilama/data'),
        Path(r'/mnt/2023_ICCV_bilama/checkpoints/bevagna')
    ]

    source_files = {p: sorted(list(p.rglob('*'))) for p in bevagna_paths}
    existing_files = {p: [] for p in server_paths}
    print('\nSource files loaded')

    for bevagna_path, server_path in zip(bevagna_paths, server_paths):
        copied_files = 0
        source_files_for_the_path = source_files[bevagna_path]
        num_files = len([f for f in source_files_for_the_path if f.is_file()])
        print(f'Processing bevagna_path: {bevagna_path}. Total files: {num_files}. Destination: {server_path}')
        for i, source_file in enumerate(source_files_for_the_path):
            if source_file.is_file():
                print(f'Processing {i+1}/{num_files}', end='\r')
                destination_file = server_path / source_file.relative_to(bevagna_path)
                if destination_file.exists():
                    existing_files[server_path].append(destination_file)
                copied_files += 1
                destination_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(str(source_file), str(destination_file))

        print(f'\nCopied {copied_files}/{num_files}')
        print(f'Num existing files: {len(existing_files[server_path])}')
        print(f'Existing files: {existing_files[server_path]} \n----------------------\n')

    str_dict = {str(key): [str(v) for v in value] for key, value in existing_files.items()}
    with open(f'/home/shared/duplicates.json', 'w') as file:
        # Use json.dump() to write the dictionary to the file
        json.dump(str_dict, file)




if __name__ == '__main__':
    main()
    sys.exit()

