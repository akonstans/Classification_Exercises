import os
import env
import pandas as pd

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
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        url = get_db_url(db_name)
        variable = pd.read_sql(query, url)
        variable.to_csv(filename)
        return variable