import yaml
import re

def read_config_file(path):
    with open(path,'r') as file:
        try:
            return yaml.safe_load(stream=file)
        except Exception as e:
            return e

def preprocessing(list):
    return [x.title() for x in list]

def validate_schema(dataframe,config_columns):

    dataframe = preprocessing(dataframe)
    config_columns =  preprocessing(config_columns)

    # Validate the number of columns
    if len(dataframe) != len(config_columns):
        print("Number of columns does not match schema.\n")

        for i in range(len(dataframe)):
            if config_columns.__contains__(dataframe[i]) == False:
                print(f"The dataframe has a column name \'{dataframe[i]}\' which does not match columns name in schema.\n")

    if len(dataframe) == len(config_columns):
        if all(col in config_columns for col in dataframe):
            if all(col in dataframe for col in config_columns):
                print('The validation process for the column schema defined in the YAML file was successful.')

    for i in range(len(config_columns)):
        if dataframe.__contains__(config_columns[i]) == False:
            print(f"The schema entry for \'{config_columns[i]}\' does not match any of the column name in dataframe.")
  
