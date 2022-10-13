import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Input:
        messages_filepath, categories_filepath: Strings with path to csv files: messages and categories
    Output:
        df: merged dataframe from messages and categories
    '''    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on="id")
    return df

def clean_data(df):
    '''
    Input:
        df: merged (single) dataframe
    Output:
        df: cleansed dataframe (new column structure for categories, no duplicates, removal of empty columns)
    ''' 
    categories = df.categories.str.split(pat=";",expand=True)
    row = categories.iloc[0]
    category_colnames = list(row.str[:-2])
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
        # replace wrong labels
        categories[column] = categories[column].replace(2,1)
    
    # remove "child_alone" category (only zero values)
    categories = categories.drop(columns=["child_alone"])
    
    # drop the original categories column from `df`
    df = df.drop(columns=["categories"])
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    
    # drop duplicates
    df = df.drop_duplicates()
    
    return df

def save_data(df, database_filename):
    '''
    Input:
        df,database_filename: cleansed dataframe, path+filename for database
    Output: None
    ''' 
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('disaster_response_data', engine, index=False, if_exists="replace") 


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
