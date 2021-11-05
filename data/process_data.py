import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Loads messages and categories ,csv files and merge them into one dataframe.

    Parameters
    ----------
    messages_filepath : string
        file path containing messages.
    categories_filepath : string
        file path containing categories for each id.

    Returns
    -------
    df : pd.DataFrame
        dataframe containing messages and categories together.

    '''
    # load messages dataset
    messages =  pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = messages.merge(categories, on='id')
    return df

def clean_data(df):
    '''
    Cleans input dataframe by dividing categories into one column per category. 
    Convert each column data into numerical and concatenate the original 
    dataframe with the categories. Drop all duplicated ids

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing messages and categories.

    Returns
    -------
    df : pd.DataFrame
        Dataframe already cleaned and with categories divided in columns.

    '''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(";",expand=True)
    # select the first row of the categories dataframe
    row = categories.loc[0]
    category_colnames = row.str.slice(0,-2,1)
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).apply(lambda row:row[-1])
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        
    # drop the original categories column from `df`
    df.drop('categories',axis=1,inplace=True)
    
    df = pd.concat([df.reset_index(),categories.reset_index()],axis=1)
    
    # drop duplicates
    df.drop_duplicates(subset='id',inplace=True)
    
    return df
        
def save_data(df, database_filename):
    '''
    Save data into a SQLite file

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe wanting to export.
    database_filename : string
        file path.

    Returns
    -------
    None.

    '''
    name = 'sqlite:///' +str(database_filename)
    engine = create_engine(name)
    df.to_sql('DisasterResponse', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()