import plotly.graph_objects as go

def modelPlot(df, date_range):
    df_plot = df.sort_values('contribution', ascending=False)
    x = df_plot['params'].values
    y = [round(i, 2) for i in df_plot['contribution'].values]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, 
                         y=y,
                         text=y,
                         textposition='outside'))
    
    fig.update_layout(title=f"MMM Model [{date_range}]",
                      width=800, height=700,
                      yaxis_title="Contribution Score",
                      yaxis_range=[min(y)-abs(max(y))/5, max(y)+abs(max(y))/5])
    
    return fig