import os
import env
import pandas as pd
from sklearn.model_selection import train_test_split


def clean(currency):
    '''Takes a currency considered an obj or str and turns it into a clean float
        rounded to two decimal places'''
    currency = currency.replace('$', '')
    currency = currency.replace(',', '')
    currency = currency.replace('-', '')
    currency = round(float(currency), 2)
    return currency

def get_db_url(db, env_file=os.path.exists('env.py')):
    '''
    return a formatted string containing username, password, host and database
    for connecting to the mySQL server
    and the database indicated
    env_file checks to see if the env.py exists in cwd
    '''
    if env_file:
        username, password, host = (env.username, env.password, env.host)
        return f'mysql+pymysql://{username}:{password}@{host}/{db}'
    else:
        return 'You need a username and password'

def connect(db_name, filename, query):
    '''
    input the db name like using the get_db_url function
    then use a filename to create a .csv file eg. 'titanic.csv'
    then write a query for what you want to select from the database

    '''
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        url = get_db_url(db_name)
        variable = pd.read_sql(query, url)
        variable.to_csv(filename)
        return variable

def train_vailidate_test_split(df, target):
    train_validate, test = train_test_split(df, train_size =.8, random_state = 91, stratify = df[target])
    train, validate = train_test_split(train_validate, train_size = .7, random_state = 91, stratify = train_validate[target])
    return train, validate, test