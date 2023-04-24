import numpy as np
import matplotlib.pyplot as plt

def add_fourier_terms(input_data, month_k):

    import numpy as np

    df = input_data.copy()

    for k in range(1, month_k+1):
        df['month_sin'+str(k)] = np.sin(2 *k* np.pi * df.index.month/12) 
        df['month_cos'+str(k)] = np.cos(2 *k* np.pi * df.index.month/12)


    return df

def decomp(serie, period, root):

    import statsmodels.api as sm
    from datetime import datetime,date, timedelta
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    rcParams['axes.linewidth'] = 1.5


    rolling_avg = serie.rolling(window=12).mean() 
    rolling_std = serie.rolling(window=12).std()


    additive_decomposition = sm.tsa.seasonal_decompose(serie,  model='additive', period=period, extrapolate_trend='freq')

    trend_estimate_ad    = additive_decomposition.trend
    periodic_estimate_ad = additive_decomposition.seasonal
    residual_ad          = additive_decomposition.resid

    fig, ax = plt.subplots(2,2, figsize=(28,14), sharex = False, sharey = False)
    #fig.suptitle(f'Decomposição Aditiva', fontsize=30)
    plt.subplot(221)
    plt.grid(False)
    plt.gca().spines[['right', 'top']].set_visible(False)
    #plt.title('Série Temporal')
    plt.xlabel('Data')
    plt.ylabel('Velocidade do Vento [m/s]')
    plt.plot(serie, label='Série Temporal Original', color='blue')
    plt.plot(rolling_avg, label='Média Móvel 12 meses', color='green')
    plt.plot(rolling_std, label='Desvio Padrão Móvel 12 meses', color='#142039')
    plt.plot(trend_estimate_ad ,label='Tendência da Série Temporal' , color='red')
    #ax[0, 0].set_xlim([date(2010, 1, 1), date(2023, 3, 31)])
    plt.legend(loc='lower left', bbox_to_anchor=(0.0, -0.15), ncol=4)
    plt.subplot(222)
    plt.grid(False)
    plt.gca().spines[['right', 'top']].set_visible(False)
    #plt.title('Trend')
    plt.xlabel('Data')
    plt.plot(trend_estimate_ad,label='Tendência da Série Temporal',color='blue')
    #ax[0, 1].set_xlim([date(2010, 1, 1), date(2023, 3, 31)])
    plt.legend(loc='lower left', bbox_to_anchor=(0.0, -0.15), ncol=1)
    plt.subplot(223)
    plt.grid(False)
    plt.gca().spines[['right', 'top']].set_visible(False)
    #plt.title('Periódico')
    plt.xlabel('Data')
    plt.plot(periodic_estimate_ad,label='Sazonalidade da Série Temporal',color='blue')
    #ax[1, 0].set_xlim([date(2010, 1, 1), date(2023, 3, 31)])
    plt.legend(loc='lower left', bbox_to_anchor=(0.0, -0.15), ncol=1)
    plt.subplot(224)
    plt.grid(False)
    plt.gca().spines[['right', 'top']].set_visible(False)
    #plt.title('Residual')
    plt.xlabel('Data')
    plt.plot(residual_ad,label='Decomposição Residual da Série Temporal', marker='o', linestyle= '', color='blue')

    plt.legend(loc='lower left', bbox_to_anchor=(0.0, -0.15), ncol=1)
    plt.tight_layout()
    plt.savefig(f'{root}/figures/Signal_decomposition_additive.png')  
    plt.show()

    return trend_estimate_ad, periodic_estimate_ad, residual_ad


def check_stationarity_plots_acf_pacf(serie, trend_estimate, periodic_estimate, residual, root):

    from statsmodels.tsa.stattools import adfuller
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    import pandas as pd
    from matplotlib import rcParams

    rcParams['axes.linewidth'] = 1.5

    dict_names_variables = {str(serie): 'Serie',
                            str(trend_estimate): 'Trend',
                            str(periodic_estimate): 'Periodic',
                            str(residual): 'Residual' 
                            }

    for variable in [serie, trend_estimate, periodic_estimate, residual]:

        print(f'Results of Augmented Dickey–Fuller Test {dict_names_variables[str(variable)]}')
        
        dftest = adfuller(variable.dropna(), autolag='AIC')
        
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

        for key,value in dftest[4].items():
            dfoutput['Critical Value (%s)'%key] = value
            
        print(dfoutput.to_string())

        df = pd.DataFrame.from_dict(dfoutput).reset_index(drop=True)


        if dfoutput[1] <= 0.05:
            print("Strong evidence against the null hypothesis")
            print("Reject the null hypothesis")
            print("Data has no unit root and is stationary")
            
            df.at[7, df.columns[0]] = "Strong evidence against the null hypothesis"
            df.at[8, df.columns[0]] = "Reject the null hypothesis"
            df.at[9, df.columns[0]] = "Data has no unit root and is stationary"

        else:
            print("Weak evidence against the null hypothesis")
            print("Fail to reject the null hypothesis")
            print("Data has a unit root and is non-stationary")

            df.at[7, df.columns[0]] = "Weak evidence against the null hypothesis"
            df.at[8, df.columns[0]] = "Fail to reject the null hypothesis"
            df.at[9, df.columns[0]] = "Data has a unit root and is non-stationary"

        df.to_csv(f'{root}/dataframe_check_stationarity_{dict_names_variables[str(variable)]}.csv', sep=',', index=True)

        del dftest, dfoutput, df

        fig, ax = plt.subplots(2,1, figsize=(14,12))
        plt.grid(False)
        

        plot_acf(variable, ax=ax[0])

        ax[0].spines[['right', 'top']].set_visible(False)

        ax[0].set_title('')
    
        plot_pacf(variable, ax=ax[1])

        ax[1].spines[['right', 'top']].set_visible(False)

        ax[1].set_title('')

        plt.tight_layout()
        plt.savefig(f'{root}/figures/acf_pcaf_{dict_names_variables[str(variable)]}.png')  
        plt.show()


def sarimax_diagnostic(auto_arima_dict, root):

    import matplotlib.pyplot as plt
    from pmdarima.arima import auto_arima
    from matplotlib import rcParams

    rcParams['axes.linewidth'] = 1.5


    order = {}

    mod = auto_arima(y = auto_arima_dict['serie'], 
                    start_p=auto_arima_dict['start_p'], start_q=auto_arima_dict['start_q'], 
                    start_P=auto_arima_dict['start_P'], start_Q=auto_arima_dict['start_Q'], 
                    max_p=auto_arima_dict['max_p'], max_q=auto_arima_dict['max_q'], 
                    max_P=auto_arima_dict['max_P'], max_Q=auto_arima_dict['max_Q'],
                    max_d=auto_arima_dict['max_d'], max_D=auto_arima_dict['max_D'],
                    max_order=auto_arima_dict['max_order'], d=auto_arima_dict['d'], D= auto_arima_dict['D'], 
                    test=auto_arima_dict['test'], m=auto_arima_dict['m'],
                    stepwise=auto_arima_dict['stepwise'], trace=auto_arima_dict['trace'],
                    stationary = auto_arima_dict['stationary'], seasonal= auto_arima_dict['seasonal']
                    ) 
               
                    ##random=True, random_state = 42)
                    ##approximation=None, method=None, truncate=None, 
                    ##test_kwargs=None, seasonal_test='seas',
                    ##seasonal_test_kwargs=None, allowdrift=True, allowmean=True,
                    ##blambda=None, biasadj=False, parallel=False, season_length=period)


    results = mod.fit(np.array(auto_arima_dict['serie']))

    print(results.summary())

    fig = results.plot_diagnostics(figsize=(16, 8))
    plt.grid(False)
    plt.gca().spines[['right', 'top']].set_visible(False)

    plt.tight_layout()

    plt.savefig(f'{root}/figures/sarimax_initial_test.png')

    plt.show()

    order[f'orders']=[results.order, results.seasonal_order]

    return order



def sarimax_crosvalidation(df, order, seasonal_order, initial, period, horizon, exog_col=None):

    from datetime import datetime,date, timedelta
    import pandas as pd

    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from dateutil.relativedelta import relativedelta
    

    cutoffs = []
    cutoff  = df.ds.min()+relativedelta(months=initial)

    while cutoff<df.ds.max():
        cutoffs=cutoffs + [cutoff]
        cutoff = cutoff+relativedelta(months=period)
    
    sarimax_cv = pd.DataFrame()

    for cutoff in cutoffs:

        train_data = df.loc[df.ds<cutoff]
        
        outsample = df.loc[(df.ds>cutoff) & (df.ds<=cutoff+relativedelta(months=horizon))]

        if len(outsample)>0:
            
            if exog_col == None:
                
                model = model = SARIMAX(train_data[['y']],
                                order=order, 
                                seasonal_order=seasonal_order, enforce_stationarity=False)
                
                results = model.fit(disp=False, maxiter=300)
                

                forecast= results.get_forecast(steps=len(outsample[['y']]), alpha=0.01)

            else:
                model = SARIMAX(train_data[['y']], 
                                order=order, 
                                seasonal_order=seasonal_order, 
                                exog=train_data[exog_col], enforce_stationarity=False)
                
                results = model.fit(disp=False, maxiter=300)

                forecast = results.get_forecast(steps=len(outsample[['y']]), alpha=0.01, exog=outsample[exog_col])

            outsample['yhat'] = forecast.predicted_mean
            outsample['yhat_upper'] = forecast.conf_int()['upper y']
            outsample['yhat_lower'] = forecast.conf_int()['lower y']
            outsample['cutoff'] = cutoff

            if exog_col == None:
                outsample = outsample[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'y', 'cutoff']]
            else:
                outsample = outsample[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'y', 'cutoff']+ exog_col]

            outsample = outsample.dropna()

            sarimax_cv = sarimax_cv.append(outsample)

    return sarimax_cv
    

def grid_search_prophet(df, initial, period, horizon, param_grid={}, exog_col=None):

    from prophet import Prophet
    
    import itertools
    import pandas as pd
    from prophet.diagnostics import cross_validation, performance_metrics
    from dateutil.relativedelta import relativedelta
    
    if param_grid=={}:
        param_grid = {  
                'changepoint_prior_scale': [0.01, 0.05, 0.1, 0.5, 1],
                'seasonality_prior_scale': [0.05, 0.1, 1.0],
                'seasonality_mode':['multiplicative','additive'],
                'weekly_seasonality':[False],
            }

    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product (*param_grid.values())]
    rmses = [] 

    cutoffs = []
    cutoff  = df.ds.min()+relativedelta(months=initial)
    while cutoff<df.ds.max()-relativedelta(months=period*5):
        cutoffs=cutoffs + [cutoff]
        cutoff = cutoff+relativedelta(months=period)

    for params in all_params:
        
        m = Prophet(**params)
        
        if exog_col!=None:
            for col in exog_col:
                m.add_regressor(col) 
        
        m.fit(df) 
        prophet_cv = cross_validation(m, cutoffs=cutoffs, horizon=horizon)
        
        df_p = performance_metrics(prophet_cv, rolling_window=1)
        
        rmses.append(df_p['rmse'].values[0])

    tuning_results = pd.DataFrame(all_params)
    tuning_results['rmse'] = rmses
    
    tuning_results.sort_values('rmse', ascending=True)
    best_params = all_params[np.argmin(rmses)]
    
    return best_params, tuning_results, df_p

def objective(trial, df, initial, period, horizon):

    from prophet import Prophet
    import itertools
    import pandas as pd
    from prophet.diagnostics import cross_validation, performance_metrics
    import contextlib
    from dateutil.relativedelta import relativedelta

    param_types = {'changepoint_prior_scale': 'float', 
                'seasonality_prior_scale': 'float',
                'seasonality_mode': 'int'}

    bounds = {'changepoint_prior_scale': [0.001, 0.5],
            'seasonality_prior_scale': [0.01, 10],
            'seasonality_mode': [0, 1]}
    params = {}
    
    for param in ['changepoint_prior_scale', 'seasonality_prior_scale']:
        params[param] = trial.suggest_uniform(param, bounds[param][0], bounds[param][1])
        
    estacionality = ['additive', 'multiplicative']
    
    params['seasonality_mode'] = estacionality[trial.suggest_int('seasonality_mode', 
        bounds['seasonality_mode'][0], 
        bounds['seasonality_mode'][1]
    )]

    m = Prophet(weekly_seasonality=False,
                daily_seasonality=False,
                **params)
    
    cols = df.drop(['ds', 'y'], axis=1).columns.values.tolist()

    if cols!=None:
        for col in cols:
            m.add_regressor(col)      

    m.fit(df) 

    cutoffs = []
    cutoff  = df.ds.min()+relativedelta(months=initial)
    while cutoff<df.ds.max()-relativedelta(months=period*5):
        cutoffs=cutoffs + [cutoff]
        cutoff = cutoff+relativedelta(months=period)


    df_cv = cross_validation(m, cutoffs=cutoffs, horizon=horizon)
    
    df_p = performance_metrics(df_cv, rolling_window=1)
    
    return df_p['rmse'].values[0]
    
def optuna_prophet(df, initial, period, horizon):
    
    import optuna

    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner(n_warmup_steps=3))
    study.optimize(lambda trial: objective(trial =trial, df =df, initial=initial, period=period, horizon=horizon), n_trials=15)
    
    best_params = study.best_params
    
    if best_params['seasonality_mode']==1:
        best_params['seasonality_mode']='multiplicative'
    else:   
        best_params['seasonality_mode']='additive'

    return best_params

def plots_prophet(m, df_cv, prefix, root):


    import statsmodels.api as sm
    from prophet import Prophet  
    import matplotlib.pyplot as plt
    from prophet.plot import plot_cross_validation_metric
    from matplotlib import rcParams
    rcParams['axes.linewidth'] = 1.5
    

    fig = m.plot(df_cv, uncertainty=True)
    plt.grid(False)
    plt.gca().spines[['right', 'top']].set_visible(False)
    #plt.title(f'{prefix} model Time Serie')
    plt.ylabel('Velocidade do Vento [m/s]')
    plt.xlabel('Data')
    plt.tight_layout()
    plt.savefig(f'{root}/figures/{prefix}_model_TS.png')
    plt.show()

    fig = plot_cross_validation_metric(df_cv, metric='rmse', rolling_window=30)
    plt.grid(False)
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.xlabel('Horizonte (dias)')
    #plt.title(f'{prefix} model RMSE')
    plt.tight_layout()
    plt.savefig(f'{root}/figures/{prefix}_model_RMSE.png')
    plt.show()

    fig = plot_cross_validation_metric(df_cv, metric='mape', rolling_window=30)
    plt.grid(False)
    plt.gca().spines[['right', 'top']].set_visible(False)
    #plt.title(f'{prefix} model MAPE')
    plt.xlabel('Horizonte (dias)')
    plt.tight_layout()
    plt.savefig(f'{root}/figures/{prefix}_model_MAPE.png')
    plt.show()

def train_pred(df, cutoff, features, ohe_features):

    import pandas as pd
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.ensemble import AdaBoostRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import Lasso
    from sklearn.tree import DecisionTreeRegressor

    treino = df.loc[(df.delta>=1) & (df.date_ref<cutoff) & (df.date_forecast<cutoff)].dropna()
    test = df[df.date_forecast==cutoff]
    
    X_treino = treino[features+ohe_features]
    X_test   = test[features+ohe_features]
    
    y_treino = treino['y']
        
    numeric_transformer = Pipeline([
        ("scaler", MinMaxScaler()),
        ('poly',PolynomialFeatures(2)),
        ("pca", PCA(n_components=3))                
        ])
            
    ohe_transformer = OneHotEncoder(drop='first',handle_unknown="ignore")
    
    if len(ohe_features)>0:
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, features),
                ("ohe", ohe_transformer, ohe_features),
            ]
        )
    else:
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, features),
            ]
        )

    pipe_lr = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    pipe_rf = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor())
    ])

    pipe_ls = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', Lasso())
    ])

    pipe_dt = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', DecisionTreeRegressor())
    ])

    pipe_gbr = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor())
    ])

    pipe_ada = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', AdaBoostRegressor())
    ])
    
    lr = LinearRegression()
    
    rf = RandomForestRegressor()

    ls = Lasso()

    dt = DecisionTreeRegressor()
    
    gbr = GradientBoostingRegressor()
    
    ada = AdaBoostRegressor()
    
    pipe_lr.fit(X_treino, y_treino)
    pipe_rf.fit(X_treino, y_treino)
    pipe_ls.fit(X_treino, y_treino)
    pipe_dt.fit(X_treino, y_treino)
    pipe_gbr.fit(X_treino, y_treino)
    pipe_ada.fit(X_treino, y_treino)
    
    test['yhat_rf'] = pipe_rf.predict(X_test)
    test['yhat_lr'] = pipe_lr.predict(X_test)
    test['yhat_ls'] = pipe_ls.predict(X_test)
    test['yhat_dt'] = pipe_dt.predict(X_test)
    test['yhat_gbr'] = pipe_gbr.predict(X_test)
    test['yhat_ada'] = pipe_ada.predict(X_test)

    return test

def metrics( g ):
    '''
    function calculate rmse mape and R2 between predicted and observed values

    Args:
    g (Dataframe) : Pandas Dataframe with columns yhat, yhat_sarimax, yhat_lr, yhat_rf, newave and y

    Returns: 
    daraframe with RMSE, R2 and MAPE for each prediction of y
    '''
    
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
    import pandas as pd

    rmse         = np.sqrt( mean_squared_error( g['y'], g['yhat_prophet'] ) )
    rmse_sarimax = np.sqrt( mean_squared_error( g['y'], g['yhat_sarimax'] ) )
    rmse_lr      = np.sqrt( mean_squared_error( g['y'], g['yhat_lr'] ) )
    rmse_rf      = np.sqrt( mean_squared_error( g['y'], g['yhat_rf'] ) )
    rmse_ls      = np.sqrt( mean_squared_error( g['y'], g['yhat_ls'] ) )
    rmse_dt      = np.sqrt( mean_squared_error( g['y'], g['yhat_dt'] ) )
    rmse_gbr      = np.sqrt( mean_squared_error( g['y'], g['yhat_gbr'] ) )
    rmse_ada      = np.sqrt( mean_squared_error( g['y'], g['yhat_ada'] ) )

    mape          =  mean_absolute_percentage_error( g['y'], g['yhat_prophet'] )
    mape_sarimax  =  mean_absolute_percentage_error( g['y'], g['yhat_sarimax'] )
    mape_lr       = mean_absolute_percentage_error( g['y'], g['yhat_lr'] ) 
    mape_rf       = mean_absolute_percentage_error( g['y'], g['yhat_rf'] )
    mape_ls      = mean_absolute_percentage_error( g['y'], g['yhat_ls'] ) 
    mape_dt      = mean_absolute_percentage_error( g['y'], g['yhat_dt'] ) 
    mape_gbr      = mean_absolute_percentage_error( g['y'], g['yhat_gbr'] ) 
    mape_ada      = mean_absolute_percentage_error( g['y'], g['yhat_ada'] ) 

    r2           = r2_score(g['y'], g['yhat_prophet'] )
    r2_sarimax   = r2_score(g['y'], g['yhat_sarimax'] )
    r2_lr        = r2_score(g['y'], g['yhat_lr'] )
    r2_rf        = r2_score(g['y'], g['yhat_rf'] )
    r2_ls      = r2_score( g['y'], g['yhat_ls'] ) 
    r2_dt      = r2_score( g['y'], g['yhat_dt'] ) 
    r2_gbr      = r2_score( g['y'], g['yhat_gbr'] ) 
    r2_ada      = r2_score( g['y'], g['yhat_ada'] ) 
    
    m = dict(           rmse = rmse,
                        rmse_sarimax = rmse_sarimax, 
                        rmse_lr= rmse_lr, 
                        rmse_rf=rmse_rf, 
                        rmse_ls=rmse_ls,
                        rmse_dt=rmse_dt,
                        rmse_gbr=rmse_gbr,
                        rmse_ada=rmse_ada,

                        r2 = r2, 
                        r2_sarimax = r2_sarimax, 
                        r2_lr=r2_lr, 
                        r2_rf=r2_rf,
                        r2_ls=r2_ls,
                        r2_dt=r2_dt,
                        r2_gbr=r2_gbr,
                        r2_ada=r2_ada, 
                        
                        mape = mape,
                        mape_sarimax = mape_sarimax, 
                        mape_lr=mape_lr, 
                        mape_rf=mape_rf,
                        mape_ls=mape_ls,
                        mape_dt=mape_dt,
                        mape_gbr=mape_gbr,
                        mape_ada=mape_ada)

    return pd.Series(m)