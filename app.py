import streamlit as st
import pandas as pd
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

from module.mmmModel import mmm_model
from module.semTransform import transforms_df
from module.semModel import sem_model


def run():
    st.set_page_config(page_title="SEM Model", 
                       page_icon="📈", 
                       layout="wide", # centered, wide
                       initial_sidebar_state="auto")
    st.title('Budget Allocation')
    
    cols = ['Date', 'Campaign_Type', 'Ad_Group']
    
    df = pd.read_csv('sem_sample_Jan_Aug_22_wdate.csv')
    df = df.rename(columns={'Ad Group': 'Ad_Group'})
    df['Campaign_Type'] = df['Campaign'].apply(lambda x: 'Branded' if 'branded keywords' in x.lower() else 'Non-branded')
    
    tabs = st.tabs(['MMM Report', 'SEM Prediction'])
    with tabs[0]:
        mmm_model(df, cols, 0.1)
    with tabs[1]:
        dl_tabs = st.columns((1,2))
        with dl_tabs[0]:
            with st.expander('Download Template'):
                sample = pd.read_csv('sample.csv')
                sample = sample.to_csv().encode('utf-8')
                st.download_button(
                    label="Download Prediction Template",
                    data=sample,
                    file_name='sample.csv',
                    mime='text/csv',
                )
        df_rev = transforms_df(df)
        smodel, le1, le2, poly_reg, score, mae, rmse = sem_model(df=df_rev)
        
        sem_cols = st.columns((1,1,1))
        with sem_cols[0]:
            st.metric('Score', score)
        with sem_cols[1]:
            st.metric('MAE', mae)
        with sem_cols[2]:
            st.metric('RMSE', rmse)
        
        uploaded_files = st.file_uploader("Upload your files", accept_multiple_files=False, type=["csv"])
        if uploaded_files:
            df_real = pd.read_csv(uploaded_files)
            df_real['Campaign_Type'] = le1.fit_transform(df_real['Campaign_Type'])
            df_real['Ad_Group'] = le2.fit_transform(df_real['Ad_Group'])
            real_input = poly_reg.fit_transform(df_real)
            y_pred = smodel.predict(real_input)
            
            df_show = df_real.copy()
            df_show['Campaign_Type'] = le1.inverse_transform(df_show['Campaign_Type'])
            df_show['Ad_Group'] = le2.inverse_transform(df_show['Ad_Group'])
            df_show['Predicted'] = [x if x>=0 else 0 for x in y_pred]
            st.write(df_show)
            st.subheader('Predicted Revenue: {}'.format(df_show['Predicted'].sum()))
            st.subheader('Predicted ROAS: {}'.format(df_show['Predicted'].sum()/df_show['Cost'].sum()))
        
    
if __name__ == "__main__":
    run()
    
