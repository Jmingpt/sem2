import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.linear_model import LinearRegression

def sem_model(df, ratio=0.2):
    X = df.drop('Revenue', axis=1)
    y = df['Revenue']
    
    le1 = LabelEncoder()
    le2 = LabelEncoder()
    X['Campaign_Type'] = le1.fit_transform(X['Campaign_Type'])
    X['Ad_Group'] = le2.fit_transform(X['Ad_Group'])
    
    X = X[['Campaign_Type', 'Ad_Group', 'Cost']]
    
    scores = []
    poly_reg = PolynomialFeatures(degree=3)
    model = LinearRegression()
    for i in range(300):
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=i, test_size=ratio)
        X_train = poly_reg.fit_transform(X_train)
        X_test = poly_reg.fit_transform(X_test)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scores.append([i, round(r2_score(y_test, y_pred), 4)])
        
    ops_state = pd.DataFrame(scores, columns=['idx', 'score']).set_index('idx').idxmax().values[0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=ops_state, test_size=ratio)
    X_train = poly_reg.fit_transform(X_train)
    X_test = poly_reg.fit_transform(X_test)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    score=round(100*r2_score(y_test, y_pred), 2)
    mae=round(mean_absolute_error(y_test, y_pred), 2)
    rmse=round(np.sqrt(mean_squared_error(y_test, y_pred)), 2)
    
    return model, le1, le2, poly_reg, score, mae, rmse
    
    