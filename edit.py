import os

folder_path = 'nu3'
i = 0
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)

    if os.path.isfile(file_path):
        new_filename = str(i) + ".jpg"
        new_file_path = os.path.join(folder_path, new_filename)

        # Rename the file
        os.rename(file_path, new_file_path)
        print(f'Renamed "{filename}" to "{new_filename}"')
        i = i + 1
