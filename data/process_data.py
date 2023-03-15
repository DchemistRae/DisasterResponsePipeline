import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    load data
    load data from csv files and merge into a single pandas dataframe

    input
    messages_filepath   filepath to messages.csv file
    categories_filepath filepath to categories.csv file

    Returns
    df  dataframe containing merged categories and messages
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = pd.merge(messages, categories, how='inner', on='id')
    return df


def clean_data(df):
    '''
    clean data
    accepts pandas dataframe with messages and categories and processes it into a clean dataframe

    input
    df  pandas dataframe containing messages and categories they can be classified into

    Returns
    df  a cleaned pandas dataframe
    '''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';', n=-1, expand=True)
    
    # select the first row of the categories dataframe and assign the names to columns
    row = categories.iloc[0] 
    category_colnames = row.apply(lambda x: x.split('-')[0])
    categories.columns = category_colnames
    
    # set each value to 1 or 0 and convert to numeric
    for column in categories:
        categories[column] = categories[column].str[-1:]
        categories[column] = pd.to_numeric(categories[column])
        
    # drop the original categories column from `df`
    df.drop('categories', axis = 'columns', inplace = True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)

    # remove rows with non binary values
    df = df[df['related'] != 2]
    
    # drop duplicates
    df.drop_duplicates(inplace = True)
    
    return df


def save_data(df, database_filename):
    '''
    load data
    load pandas dataframe and save into a SQLite database table

    input
    df  pandas dataframe to be saved
    database_filename   name for creating the SQLite database

    Return
    None
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('processed_data', engine, if_exists='replace',index=False)
    pass  


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