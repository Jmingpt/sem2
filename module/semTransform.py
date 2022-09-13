import pandas as pd


def transforms_df(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df_rev = df[['Date', 'Campaign_Type', 'Ad_Group', 'Cost', 'Revenue']].copy()
    df_rev = df_rev.groupby(['Date', 'Campaign_Type', 'Ad_Group']).sum().reset_index().set_index('Date')
    
    return df_rev