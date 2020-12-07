def find_nonnumeric_columns(data):
    """
    Finds the columns in a dataframe that contain ANY non-numeric values, and returns those column names as a list
    """
    columns_with_string_type = data.applymap(lambda x: isinstance(x, str))
    column_check = columns_with_string_type.any(axis=0) #.reset_index()
    return column_check[column_check == True].index.tolist()


def print_something():
    print("something")