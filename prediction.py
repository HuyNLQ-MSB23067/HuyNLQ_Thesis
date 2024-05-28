# !pip3 install pmdarima
# !pip3 install arch
# !pip3 install pyyaml h5py
# !pip3 install matplotlib

# Configuration
MAX_DATA_HOUR = 672
MAX_DATA_HOUR_PLOT = 96
MAX_DATA_DAY = -1
MAX_DATA_DAY_PLOT = -1

# Data processing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from sklearn import preprocessing
import pickle
import sys
import os

# Arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf # for determining (p,q) orders
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tools.eval_measures import mse,rmse
from sklearn.metrics import mean_absolute_percentage_error as maperror    # for ETS Plots
from pmdarima import auto_arima
from sklearn import metrics

# GARCH
from matplotlib import pyplot
from arch import arch_model

print('Starting')
sys.stdout.flush()

def process_data(filepath, id):
    # Get data
    origin_df = pd.read_csv(filepath, encoding = 'utf-8')

    print('Filtering')
    sys.stdout.flush()

    # Filter and sorting
    df = origin_df[['SESSION_ID', 'TOTAL_TRANSACTIONS', 'TOTAL_TRANSACTIONS_GMV', 'TIMESTAMP_UTC']]
    df.sort_values(by=['TIMESTAMP_UTC'])
    df['timestamp'] = pd.to_datetime(df['TIMESTAMP_UTC'])

    # Initialize Session and Transaction Counts
    df['SESSION_COUNT'] = 1
    df.loc[df['TOTAL_TRANSACTIONS'] > 0, 'TOTAL_TRANSACTIONS'] = 1

    # ==== Grouping Data ====

    print('Grouping')
    sys.stdout.flush()

    # Hourly
    df['timestamp_data'] = df['timestamp'].dt.floor('H')
    df_hour = df.groupby([df['timestamp'].dt.date, df['timestamp'].dt.hour]).agg({
        'TOTAL_TRANSACTIONS': 'sum',
        'SESSION_COUNT': 'sum',
        'timestamp_data':'first',
    })
    df_hour['CVR'] = df_hour['TOTAL_TRANSACTIONS'] / df_hour['SESSION_COUNT'] * 100
    df_hour.index = df_hour['timestamp_data']

    # Daily
    df_day = df.groupby([df['timestamp'].dt.date]).agg({
        'TOTAL_TRANSACTIONS': 'sum',
        'SESSION_COUNT': 'sum',
        'timestamp_data':'first',
    })
    df_day['CVR'] = df_day['TOTAL_TRANSACTIONS'] / df_day['SESSION_COUNT'] * 100
    df_day['day'] = df_day.index

    # ==== Cleaning and Indexing Data ====

    print('Cleaning')
    sys.stdout.flush()

    # Hourly
    df_hour["hour"] = df_hour['timestamp_data'].dt.hour
    df_hour = df_hour.rename(columns = {
        'TOTAL_TRANSACTIONS':'total_transactions',
        'SESSION_COUNT':'session_count',
        'timestamp_data':'timestamp'
    })

    # Daily
    df_day = df_day.rename(columns = {
        'TOTAL_TRANSACTIONS':'total_transactions',
        'SESSION_COUNT':'session_count'
    })

    # ==== Exporting ====

    print('Exporting')
    sys.stdout.flush()
    outdir = f'./uploads/results/{id}'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    df_hour.to_csv(f'./uploads/results/{id}/hourly_data.csv', index=True)
    df_day.to_csv(f'./uploads/results/{id}/daily_data.csv', index=True)

    # ==== Using Models ====

    print('Building')
    sys.stdout.flush()

    # --- ARIMA ---

    # ++ Hourly ++
    # train_arima_model(df=df_hour,
    #     max_train=MAX_DATA_HOUR,
    #     target_data='session_count',
    #     period_type='hourly',
    #     m=24,
    #     id=id)

    # train_arima_model(df=df_hour,
    #     max_train=MAX_DATA_HOUR,
    #     target_data='total_transactions',
    #     period_type='hourly',
    #     m=24,
    #     id=id)

    # export_prediction(target_data='session_count',
    #     period_type='hourly',
    #     plot_title='Session Count Predition by Hour',
    #     plot_value='Session Count',
    #     max_plot_size=MAX_DATA_HOUR_PLOT,
    #     freq='h',
    #     model='A',
    #     id=id)

    # export_prediction(target_data='session_count',
    #     period_type='hourly',
    #     plot_title='Session Count Prediction by Hour (ARIMA-GARCH)',
    #     plot_value='Session Count',
    #     max_plot_size=MAX_DATA_HOUR_PLOT,
    #     freq='h',
    #     model='AG',
    #     id=id)

    # export_prediction(target_data='total_transactions',
    #     period_type='hourly',
    #     plot_title='Transaction Count Predition by Hour',
    #     plot_value='Transaction Count',
    #     max_plot_size=MAX_DATA_HOUR_PLOT,
    #     freq='h',
    #     model='A',
    #     id=id)

    # export_prediction(target_data='total_transactions',
    #     period_type='hourly',
    #     plot_title='Transaction Count Prediction by Hour (ARIMA-GARCH)',
    #     plot_value='Transaction Count',
    #     max_plot_size=MAX_DATA_HOUR_PLOT,
    #     freq='h',
    #     model='AG',
    #     id=id)

    # export_prediction(target_data='CVR',
    #     period_type='hourly',
    #     plot_title='CVR Prediction by Hour (ARIMA)',
    #     plot_value='CVR',
    #     max_plot_size=MAX_DATA_HOUR_PLOT,
    #     freq='h',
    #     model='A',
    #     id=id)

    # export_prediction(target_data='CVR',
    #     period_type='hourly',
    #     plot_title='CVR Prediction by Hour (ARIMA-GARCH)',
    #     plot_value='CVR',
    #     max_plot_size=MAX_DATA_HOUR_PLOT,
    #     freq='h',
    #     model='AG',
    #     id=id)

    # ++ Daily ++
    train_arima_model(df=df_day,
        max_train=MAX_DATA_DAY,
        target_data='session_count',
        period_type='daily',
        m=7,
        id=id)

    train_arima_model(df=df_day,
        max_train=MAX_DATA_DAY,
        target_data='total_transactions',
        period_type='daily',
        m=7,
        id=id)

    export_prediction(target_data='session_count',
        period_type='daily',
        plot_title='Session Count Predition by Hour',
        plot_value='Session Count',
        max_plot_size=MAX_DATA_DAY_PLOT,
        freq='D',
        model='A',
        id=id)

    export_prediction(target_data='session_count',
        period_type='daily',
        plot_title='Session Count Prediction by Day (ARIMA-GARCH)',
        plot_value='Session Count',
        max_plot_size=MAX_DATA_DAY_PLOT,
        freq='D',
        model='AG',
        id=id)

    export_prediction(target_data='total_transactions',
        period_type='daily',
        plot_title='Transaction Count Predition by Day',
        plot_value='Transaction Count',
        max_plot_size=MAX_DATA_DAY_PLOT,
        freq='D',
        model='A',
        id=id)

    export_prediction(target_data='total_transactions',
        period_type='daily',
        plot_title='Transaction Count Prediction by Day (ARIMA-GARCH)',
        plot_value='Transaction Count',
        max_plot_size=MAX_DATA_DAY_PLOT,
        freq='D',
        model='AG',
        id=id)

    export_prediction(target_data='CVR',
        period_type='daily',
        plot_title='CVR Prediction by Day (ARIMA)',
        plot_value='CVR',
        max_plot_size=MAX_DATA_DAY_PLOT,
        freq='D',
        model='A',
        id=id)

    export_prediction(target_data='CVR',
        period_type='daily',
        plot_title='CVR Prediction by Day (ARIMA-GARCH)',
        plot_value='CVR',
        max_plot_size=MAX_DATA_DAY_PLOT,
        freq='D',
        model='AG',
        id=id)

def train_arima_model(df, max_train, target_data, period_type, m, id):
    # Training Model
    print('Building ' + period_type + ' ' + target_data)
    sys.stdout.flush()
    if max_train > 0:
        df_input = df[:-max_train]
    else:
        df_input = df
        
    # Split train / test data by 8 : 2 ratio
    test_size = int(np.floor(len(df_input) / 5))

    df_target = df_input[target_data]
    train, test = df_target.iloc[:-test_size], df_target.iloc[-test_size:]
    x_train, x_test = np.array(range(train.shape[0])), np.array(range(train.shape[0], df_target.shape[0]))

    file_path = f"./uploads/results/{id}/{period_type}_{target_data}_A_model.pkl"
    # Check if the file exists before opening it
    if os.path.exists(file_path):
        with open(file_path, 'rb') as pkl:
            arima_model = pickle.load(pkl)
    else:
        arima_model = auto_arima(train, start_p=1, start_q=1,
                              test='adf',
                              max_p=5, max_q=5,
                              m=m,
                              d=1,
                              seasonal=True,
                              start_P=0,
                              D=None,
                              trace=True,
                              error_action='ignore',
                              suppress_warnings=True,
                              stepwise=True)
        with open(f"./uploads/results/{id}/" + period_type + "_" + target_data + "_A_model.pkl", 'wb') as pkl:
            pickle.dump(arima_model, pkl)

    # Forecast
    print('Forecasting ' + period_type + ' ' + target_data)
    sys.stdout.flush()
    prediction, confint = arima_model.predict(n_periods=test_size, return_conf_int=True)
    prediction_series = pd.Series(prediction,index=test.index)
    MAE = str(metrics.mean_absolute_error(test, prediction_series))
    MSE = str(metrics.mean_squared_error(test, prediction_series))
    RMSE = str(np.sqrt(metrics.mean_squared_error(test, prediction_series)))

    pd.DataFrame(prediction).to_csv(f"./uploads/results/{id}/" + period_type + "_" + target_data + "_A_prediction_test.csv")

    if period_type == 'hourly':
        n_periods = 48
    else:
        n_periods = 14

    # Now add the actual samples to the model and create NEW forecasts
    arima_model.update(test)
    new_prediction, new_confint = arima_model.predict(n_periods=n_periods, return_conf_int=True)

    pd.DataFrame(new_prediction).to_csv(f"./uploads/results/{id}/" + period_type + "_" + target_data + "_A_prediction_final.csv")

    # To-do: Save MSE/RMSE


    # APPLY GARCH

    # fit ARIMA on returns 
    p, d, q = arima_model.order
    arima_residuals = arima_model.arima_res_.resid

    # GARCH
    min_aic = -1.0
    min_aic_p = -1
    min_aic_q = -1
    for p in range(1, 5):
      for q in range(1, 5):
        arima_garch_model = arch_model(arima_residuals, p=p, q=q)
        arima_garch_model_fit = arima_garch_model.fit()
        if (min_aic == -1 or min_aic > arima_garch_model_fit.aic):
          min_aic = arima_garch_model_fit.aic
          min_aic_p = p
          min_aic_q = q
          best_arima_garch_model = arima_garch_model
    garch_fitted = best_arima_garch_model.fit()

    actual_values = test

    arima_garch_prediction_series = []
    for i in range(len(actual_values)):
        # Use ARIMA to predict mu
        predicted_mu = arima_model.predict(n_periods=i+1)[-1]
        # Use GARCH to predict the residual
        garch_forecast = garch_fitted.forecast(horizon=i+1)
        predicted_et = garch_forecast.mean['h.' + str(i+1)].iloc[-1]
        # Combine both models' output: yt = mu + et
        prediction = predicted_mu + predicted_et
        arima_garch_prediction_series.append(prediction)

    print("MAE: " + str(metrics.mean_absolute_error(test, arima_garch_prediction_series)))
    sys.stdout.flush()
    print("MSE: " + str(metrics.mean_squared_error(test, arima_garch_prediction_series)))
    sys.stdout.flush()
    print("RMSE: " + str(np.sqrt(metrics.mean_squared_error(test, arima_garch_prediction_series))))
    sys.stdout.flush()

    arima_garch_prediction_series = arima_garch_prediction_series[:n_periods]

    pd.DataFrame(arima_garch_prediction_series).to_csv(f"./uploads/results/{id}/" + period_type + "_" + target_data + "_AG_prediction_final.csv")

def export_prediction(target_data, period_type, plot_title, plot_value, max_plot_size, freq, model, id):
    original_df = pd.read_csv(f'./uploads/results/{id}/' + period_type + '_data.csv', encoding = 'utf-8')
    # prediction_df = pd.read_csv(f'' + period_type + '_prediction_final.csv', encoding = 'utf-8')
    if target_data == 'CVR':
        prediction_T_df = pd.read_csv(f'./uploads/results/{id}/' + period_type + '_total_transactions_' + model + '_prediction_final.csv', index_col=0)
        prediction_S_df = pd.read_csv(f'./uploads/results/{id}/' + period_type + '_session_count_' + model + '_prediction_final.csv', index_col=0)

        # Extracting the values for division                
        transactions_values = prediction_T_df.iloc[:, 0].values
        sessions_values = prediction_S_df.iloc[:, 0].values

        result_values = transactions_values / sessions_values * 100
        prediction_df = pd.DataFrame(result_values)
        prediction_df.to_csv(f'./uploads/results/{id}/' + period_type + "_" + target_data + '_' + model + '_prediction_final.csv')
    
    prediction_df = pd.read_csv(f'./uploads/results/{id}/' + period_type + "_" + target_data + '_' + model + '_prediction_final.csv', header=None, index_col=0)
    prediction_df.columns = [target_data]
    prediction_df = prediction_df[1:]

    original_df.index = pd.to_datetime(original_df['timestamp_data'])
    original_df = original_df[:MAX_DATA_HOUR_PLOT]

    # Adjust the prediction data index to start right after the original data's last timestamp
    last_timestamp = original_df.index[-1]
    new_timestamps = pd.date_range(start=last_timestamp, periods=len(prediction_df) + 1, freq=freq)[1:]
    prediction_df.index = new_timestamps

    last_row = original_df.iloc[[-1]][['timestamp_data', target_data]]
    prediction_df = pd.concat([last_row, prediction_df])

    original_df.index = pd.to_datetime(original_df['timestamp_data']).dt.tz_convert('Asia/Seoul')
    prediction_df.index = prediction_df.index.tz_convert('Asia/Seoul')

    # # Create a new index for the prediction data
    # new_index = range(prediction_start_index, prediction_start_index + len(prediction_df))
    # prediction_df.index = new_index
    plt.figure(figsize=(12, 6))
    plt.plot(original_df.index, original_df[target_data], label='Original Data', color='blue')

    # Plot the prediction data
    plt.plot(prediction_df.index, prediction_df[target_data], label='Prediction Data', color='red')

    pd.DataFrame(original_df).to_csv(f"./uploads/results/{id}/" + period_type + "_plot_org.csv")
    pd.DataFrame(prediction_df).to_csv(f"./uploads/results/{id}/" + period_type + "_plot_pred.csv")

    # Adding titles and labels
    plt.title(plot_title)
    plt.xlabel('Timestamp')
    plt.ylabel(plot_value)
    plt.legend()

    # Show the plot
    # plt.show()

    # Save tne plot
    plt.savefig(f'./uploads/results/{id}/' + period_type + '_' + target_data + '_' + model + '.png')    

# GARCH
def train_garch_model(df, max_train, target_data, period_type, freq):
    # Training Model
    print('Building GARCH: ' + period_type + ' ' + target_data)
    sys.stdout.flush()
    df_input = df[:-max_train]
    # Split train / test data by 8 : 2 ratio
    test_size = int(np.floor(len(df_input) / 5))

    df_target = df_input[target_data]
    train, test = df_target.iloc[:-test_size], df_target.iloc[-test_size:]

    min_aic = -1.0
    min_aic_p = -1
    min_aic_q = -1
    for p in range(1, 5):
        for q in range(1, 5):
            garch_model = arch_model(train, vol='GARCH', p=p, q=q)
            garch_model_fit = garch_model.fit()
        if (min_aic == -1 or min_aic > garch_model_fit.aic):
            min_aic = garch_model_fit.aic
            min_aic_p = p
            min_aic_q = q
            best_garch_model = garch_model

    best_garch_model_fit = best_garch_model.fit(disp='off')

    garch_yhat = best_garch_model_fit.forecast(horizon=test_size)
    garch_forecast_volatility = np.sqrt(garch_yhat.variance.values[-1, :])

    # Calculate MAE and RMSE
    mae = metrics.mean_absolute_error(test, garch_forecast_volatility)
    mse = metrics.mean_squared_error(test, garch_forecast_volatility)
    rmse = np.sqrt(mse)

    print("MAE: " + str(mae))
    sys.stdout.flush()
    print("MSE: " + str(mse))
    sys.stdout.flush()
    print("RMSE: " + str(rmse))
    sys.stdout.flush()

    # Update
    garch_model = arch_model(df_target, vol='GARCH', p=min_aic_p, q=min_aic_q)
    garch_model_fit = garch_model.fit(disp='off')

    # one-step out-of sample forecast
    garch_forecast = garch_model_fit.forecast(horizon=48)
    predicted_et = garch_forecast.mean['h.1'].iloc[-1]


# MAIN
if __name__ == "__main__":
    file_path = sys.argv[1]
    record_id = sys.argv[2]
    process_data(file_path, record_id)

# Hourly train

# train_arima_model(df=df_hour,
#     max_train=MAX_DATA_HOUR,
#     target_data='session_count',
#     period_type='hourly',
#     m=24)

# train_arima_model(df=df_day,
#     max_train=-1,
#     target_data='session_count',
#     period_type='daily',
#     m=7)

# export_prediction(target_data='session_count',
#     period_type='hourly',
#     plot_title='Session Count Predition by Hour',
#     plot_value='Session Count',
#     max_plot_size=MAX_DATA_HOUR_PLOT,
#     freq='h',
#     model='A')

# export_prediction(target_data='session_count',
#     period_type='hourly',
#     plot_title='Session Count Prediction by Hour (ARIMA-GARCH)',
#     plot_value='Session Count',
#     max_plot_size=MAX_DATA_HOUR_PLOT,
#     freq='h',
#     model='AG')

# train_arima_model(df=df_hour,
#     max_train=MAX_DATA_HOUR,
#     target_data='total_transactions',
#     period_type='hourly',
#     m=24)

# export_prediction(target_data='total_transactions',
#     period_type='hourly',
#     plot_title='Transaction Count Predition by Hour',
#     plot_value='Transaction Count',
#     max_plot_size=MAX_DATA_HOUR_PLOT,
#     freq='h',
#     model='A')

# export_prediction(target_data='CVR',
#     period_type='hourly',
#     plot_title='CVR Predition by Hour',
#     plot_value='CVR',
#     max_plot_size=MAX_DATA_HOUR_PLOT,
#     freq='h',
#     model='A')

