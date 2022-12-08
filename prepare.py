import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from env import get_connection
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

df_google.info()

df_google.describe()

df_google['Sex'].unique()

df_google['Embarked'].unique()

df_excel = pd.read_excel('titanic.xlsx')
df_excel.head(3)

df_excel_sample = df_excel.head(100)
df_excel.shape

df_excel_sample.columns[:5]

df_excel_sample.info()
df_excel_sample.select_dtypes(include=object).columns

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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from acquire import get_titanic_data

# Overfitting:
# 
# Train/Validate/Test
# 
# Model Training
# 
# When a model is overfit, it makes impeccable guesses on the training dataset.
# 
# An overfit model makes poor (relatively) predictions on out-of-sample data.
# 
# Out-of-sample data can be the validation and test sets. 

df = get_titanic_data(get_connection)
df.head()

df.head()

numerical_cols = df.select_dtypes(exclude='object').columns.to_list()

for col in numerical_cols:
    plt.hist(df[col])
    plt.title(col)
    plt.show()
categorical_cols = df.select_dtypes(include='object').columns.to_list()

for col in categorical_cols:
    print(df[col].value_counts())
    print('-----------------')
    

# There is duplicate information
# 
# Get rid of class and embarked (duplicates).
# Drop deck column for excess null values.
# Drop passenger_id because it is not helpful.
# Drop age column because it may be difficult to impute.
# 

df1 = df.drop(columns=['class','embarked', 'passenger_id', 'deck', 'age'])

df1

df1['embark_town'].value_counts()

df1['embark_town'].fillna('Southampton', inplace=True)

dummies = pd.get_dummies(df[['sex', 'embark_town']], drop_first=[True])

df = pd.concat([df, dummies], axis=1)

df
def clean_titanic(df):
    df.drop(columns=['class','embarked', 'passenger_id', 'deck', 'age', 'Unnamed: 0'], inplace=True)
    
    df['embark_town'].fillna('Southampton', inplace=True)
    
    dummies = pd.get_dummies(df[['sex', 'embark_town']], drop_first=[True])
    
    df = pd.concat([df, dummies], axis=1)
    
    return df

test_df = get_titanic_data(get_connection)
test_df.head()

clean_df = clean_titanic(test_df)
clean_df.head()

seed = 42

train, test = train_test_split(df, train_size=.7, random_state=seed, stratify=df['survived'])

train.shape, test.shape

train.head()

seed = 42

train, val_test = train_test_split(df, train_size=.7, random_state=seed, stratify=df['survived'])

validate, test = train_test_split(val_test, train_size=.5, random_state=seed, stratify=val_test['survived'])

train.shape, validate.shape, test.shape

def train_val_test(df):
    seed = 42 
    train, val_test = train_test_split(df, train_size=.7, random_state=seed, stratify=df['survived'])
    
    validate, test = train_test_split(val_test, train_size=.5, random_state=seed, stratify=val_test['survived'])
    
    return train, validate, test

imputer = SimpleImputer(strategy='most_frequent')

impute_df = get_titanic_data(get_connection)
impute_df

train, validate, test = train_val_test(impute_df)

train.shape

imputer.fit(train[['embark_town']])

train['embark_town'].isna().sum()

train['embark_town'] = imputer.transform(train[['embark_town']])

train['embark_town'].isna().sum()

# Use the function defined in acquire.py to load the iris data.
# 
# Drop the species_id and measurement_id columns.
# 
# Rename the species_name column to just species.
# 
# Create dummy variables of the species name and concatenate onto the iris dataframe. (This is for practice, we don't always have to encode the target, but if we used species as a feature, we would need to encode it).
# 
# Create a function named prep_iris that accepts the untransformed iris data, and returns the data with the transformations above applied.

from acquire import get_iris_data
iris = get_iris_data(get_connection)
iris.head()

iris.drop(columns=['species_id', 'measurement_id', 'Unnamed: 0'], inplace=True)

iris.rename(columns={'species_name':'species'}, inplace=True)
iris.head()

iris_dummies = pd.get_dummies(iris[['species']])
iris = pd.concat([iris, iris_dummies], axis=1)

iris.head()

def prep_iris(iris):
    iris.drop(columns=['species_id', 'measurement_id', 'Unnamed: 0'], inplace=True)
    
    iris.rename(columns={'species_name':'species'}, inplace=True)
    
    iris_dummies = pd.get_dummies(iris[['species']])
    iris = pd.concat([iris, iris_dummies], axis=1)
    
    return iris

# Use the function defined in acquire.py to load the Titanic data.
# 
# Drop any unnecessary, unhelpful, or duplicated columns.
# 
# Encode the categorical columns. Create dummy variables of the categorical columns and concatenate them onto the dataframe.
# 
# Create a function named prep_titanic that accepts the raw titanic data, and returns the data with the transformations above applied.

titanic = get_titanic_data(get_connection)
titanic.head()

titanic.drop(columns=['Unnamed: 0', 'passenger_id', 'age', 'embarked', 'class', 'deck'], inplace=True)
titanic.head()

titanic_dummies = pd.get_dummies(titanic[['sex', 'embark_town']], drop_first=True)
titanic = pd.concat([titanic, titanic_dummies], axis=1)
titanic

def prep_titanic(titanic):
    titanic.drop(columns=['class','embarked', 'passenger_id', 'deck', 'age', 'Unnamed: 0'], inplace=True)
    
    titanic_dummies = pd.get_dummies(titanic[['sex', 'embark_town']], drop_first=True)
    titanic = pd.concat([titanic, titanic_dummies], axis=1)
    
    return titanic

# Use the function defined in acquire.py to load the Telco data.
# 
# Drop any unnecessary, unhelpful, or duplicated columns. This could mean dropping foreign key columns but keeping the corresponding string values, for example.
# 
# Encode the categorical columns. Create dummy variables of the categorical columns and concatenate them onto the dataframe.
# 
# Create a function named prep_telco that accepts the raw telco data, and returns the data with the transformations above applied.

from acquire import get_telco_data

telco = get_telco_data(get_connection)
telco.head()

telco.drop(columns=['Unnamed: 0', 'payment_type_id', 'contract_type_id', 'internet_service_type_id', 'customer_id'], inplace=True)
telco.head()

telco_dummies = pd.get_dummies(telco[['gender', 'partner', 'dependents', 
                                      'phone_service', 'multiple_lines', 
                                      'online_security', 'online_backup', 
                                      'device_protection', 'tech_support', 
                                      'streaming_tv', 'streaming_movies', 
                                      'paperless_billing', 'churn', 'internet_service_type', 
                                      'contract_type', 'payment_type']], drop_first=True)

telco_dummies

telco = pd.concat([telco, telco_dummies], axis=1)
telco.head()

def prep_telco(telco):
    telco.drop(columns=['Unnamed: 0', 'payment_type_id', 'contract_type_id', 
                        'internet_service_type_id', 'customer_id'], inplace=True)

    
    telco_dummies = pd.get_dummies(telco[['gender', 'partner', 'dependents', 
                                      'phone_service', 'multiple_lines', 
                                      'online_security', 'online_backup', 
                                      'device_protection', 'tech_support', 
                                      'streaming_tv', 'streaming_movies', 
                                      'paperless_billing', 'churn', 'internet_service_type', 
                                      'contract_type', 'payment_type']], drop_first=True)
    
    telco = pd.concat([telco, telco_dummies], axis=1)
    
    return telco
​# Write a function to split your data into train, test and validate datasets. Add this function to prepare.py.
# 
# Run the function in your notebook on the Iris dataset, returning 3 datasets, train_iris, validate_iris and test_iris.
# 
# Run the function on the Titanic dataset, returning 3 datasets, train_titanic, validate_titanic and test_titanic.
# 
# Run the function on the Telco dataset, returning 3 datasets, train_telco, validate_telco and test_telco.

def train_val_test(df, col):
    seed = 42 
    train, val_test = train_test_split(df, train_size=.7, random_state=seed, stratify=df[col])
    
    validate, test = train_test_split(val_test, train_size=.5, random_state=seed, stratify=val_test[col])
    
    return train, validate, test

train_iris, val_iris, test_iris = train_val_test(iris, 'species')
train_iris.shape, val_iris.shape, test_iris.shape

train_titanic, val_titanic, test_titanic = train_val_test(titanic, 'survived')
train_titanic.shape, val_titanic.shape, test_titanic.shape

train_telco, val_telco, test_telco = train_val_test(telco, 'gender')
train_telco.shape, val_telco.shape, test_telco.shape