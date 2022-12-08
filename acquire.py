import new_lib as nl
import pandas as pd
import os

def get_titanic_data(get_db_url):
    if os.path.isfile('titanic.csv'):
        return pd.read_csv('titanic.csv')
    else:
        url = get_db_url('titanic_db')
        query = '''SELECT * FROM passengers'''
        df = pd.read_sql(query, url)
        df.to_csv('titanic.csv')
        return df

def get_iris_data(get_db_url):
    
    if os.path.isfile('iris.csv'):
        
        return pd.read_csv('iris.csv')
    
    else:
        url = get_db_url('iris_db')
        query = '''
                SELECT * FROM measurements 
                JOIN species USING(species_id)
                '''
        df = pd.read_sql(query, url)
        df.to_csv('iris.csv')
        return df

def get_telco_data(get_db_url):
    
    if os.path.isfile('telco.csv'):
        
        return pd.read_csv('telco.csv')
    
    else:
        url = get_db_url('telco_churn')
        query = '''SELECT * FROM customers
                    JOIN internet_service_types USING(internet_service_type_id)
                    JOIN contract_types USING(contract_type_id)
                    JOIN payment_types USING(payment_type_id)
                    '''
        df = pd.read_sql(query, url)
        df.to_csv('telco.csv')
        return df