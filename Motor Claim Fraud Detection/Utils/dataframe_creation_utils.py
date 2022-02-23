import pandas as pd


# Create pandas dataframe through csv file
def create_dataframe_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    return df


# Create dataframe from text file
def create_dataframe_from_text_file(text_file, sep):
    df = pd.read_csv(text_file, sep=sep, header=None)
    return df


# Create dataframe from dictionary
def create_dataframe_from_dictionary(input_dict):
    df = pd.DataFrame(input_dict)
    return df


# Create dataframe from excel file
def create_dataframe_from_excel(excel_file, sheet_name):
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    return df


if __name__ == '__main__':
    pass
