import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import new_lib as nl
import os
from pydataset import data

def get_titanic(get_con_func):
    
    if os.path.isfile('titanic.csv'):
        
        return pd.read_csv('titanic.csv')
    
    else:
        url = get_con_func('titanic_db')
        query = '''SELECT * FROM passengers'''
        df = pd.read_sql(query, url)
        df.to_csv('titanic.csv')
        return df

pd.read_excel('titanic.xlsx')

pd.read_clipboard()
# ## Exercises

db_iris = data('iris')
db_iris = pd.DataFrame(db_iris)
db_iris.head(3)

db_iris.shape

db_iris.columns

db_iris.info()

db_iris.describe()

sheet_url = 'https://docs.google.com/spreadsheets/d/1Uhtml8KY19LILuZsrDtlsHHDC9wuDGUSe8LTEwvdI5g/edit#gid=341089357'
csv_export_url = sheet_url.replace('/edit#gid=', '/export?format=csv&gid=')
df_google = pd.read_csv(csv_export_url)
df_google.head(3)

df_google.shape

df_google.columns

df_google['Sex'].unique()

df_google['Embarked'].unique()

df_excel = pd.read_excel('titanic.xlsx')
df_excel.head(3)

df_excel_sample = df_excel.head(100)
df_excel.shape

df_excel_sample.columns[:5]

df_excel_min_max = pd.DataFrame(columns=['max','min'])
df_excel_min_max['max'] = df_excel.select_dtypes(exclude=object).max()
df_excel_min_max['min'] = df_excel.select_dtypes(exclude=object).min()
df_excel_min_max

from acquire import get_titanic_data

get_titanic_data(get_connection).head()

from acquire import get_iris_data

get_iris_data(get_connection).head()

from acquire import get_telco_data

get_telco_data(get_connection).head()


