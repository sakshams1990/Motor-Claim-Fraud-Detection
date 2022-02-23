import os

from Utils.dataframe_creation_utils import create_dataframe_from_csv, create_dataframe_from_text_file, \
    create_dataframe_from_excel
from get_values_from_config_file import root_path, raw_dataset_folder, dataset_folder, target_column


def create_dataframe_from_source_file(input_path=None):
    input_df = None

    if root_path == "":
        print("The root path is empty!!")
    else:
        dataset_path = os.path.join(root_path, dataset_folder)
        raw_input_path = os.path.join(dataset_path, raw_dataset_folder)

        file_count = sum(len(files) for _, _, files in os.walk(raw_input_path))

        file_in_raw_input_path = None
        for _, _, files in os.walk(raw_input_path):
            file_in_raw_input_path = files

        if file_count == 0:
            print(f'No input file present in {raw_dataset_folder} folder.Please add a data source!!')
        elif file_count > 1:
            print(f'Kindly check your {raw_dataset_folder} folder. It should contain only one data source file!!')
        else:
            input_file_path = os.path.join(raw_input_path, file_in_raw_input_path[0])

            file_extension = os.path.splitext(input_file_path)[1]

            if file_extension == ".csv":
                input_df = create_dataframe_from_csv(input_file_path)
            elif file_extension == ".txt":
                input_df = create_dataframe_from_text_file(input_file_path, ";")
            elif file_extension in ['.XLSX', '.XLS', '.xls', '.xlsx']:
                input_df = create_dataframe_from_excel(input_file_path, sheet_name="export")
            else:
                print(
                    f'File with extension {file_extension} is invalid!! Try with files with extensions of csv, txt, XLSX, xls, xlsx or XLS!!')
    return input_df


def create_target_variable_mapping(input_df):
    labels = input_df[target_column].astype('category').cat.categories.tolist()
    label_map = {k: v for k, v in zip(labels, list(range(0, len(labels))))}
    return labels, label_map


if __name__ == '__main__':
    pass
