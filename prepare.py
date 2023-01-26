import pandas as pd

def prep_titanic(df):
    '''
    This is used to prepare the titanic dataset to work with. It will drop unneeded columns, as well as create and then 
    concatenate dummies onto the DataFrame
    '''
    to_drop = ['Unnamed: 0', 'class', 'embarked', 'passenger_id', 'deck']
    df.drop(columns=to_drop, inplace=True)
    
    dummies = pd.get_dummies(df[['sex', 'embark_town']], drop_first=True)
    df = pd.concat([df, dummies], axis=1)
    
    df = df.drop(columns=['sex', 'embark_town'])
    
    return df