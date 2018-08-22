from prometheus import Prometheus
import pandas
import numpy as np
from numpy import fft
import json
import time
# from lib.model import *
from ceph import CephConnect as cp
from datetime import datetime, timedelta
from fbprophet import Prophet
from sortedcontainers import SortedDict
import os
import gc
import pickle
import collections
from scipy.stats import norm

# Plotting
# import matplotlib.pyplot as plt


def get_df_from_json(metric, metric_dict_pd={}, data_window=5):
    '''
    Method to convert a json object of a Prometheus metric to a dictionary of shaped Pandas DataFrames

    The shape is dict[metric_metadata] = Pandas Object

    Pandas Object = timestamp, value
                    15737933, 1
                    .....

    This method can also be used to update an existing dictionary with new data
    '''
    # metric_dict = {}
    current_time = datetime.now()
    earliest_data_time = current_time - timedelta(days = data_window)


    print("Pre-processing Data...........")
    # metric_dict_pd = {}
    # print("Length of metric: ", len(metric))
    for row in metric:
        # metric_dict[str(row['metric'])] = metric_dict.get(str(row['metric']),[]) + (row['values'])
        metric_metadata = str(SortedDict(row['metric']))[11:-1] # Sort the dictionary and then convert it to string so it can be hashed
        # print(metric_metadata)
        # print("Row Values: ",row['values'])
        if  metric_metadata not in metric_dict_pd:
            metric_dict_pd[metric_metadata] = pandas.DataFrame(row['values'], columns=['ds', 'y']).apply(pandas.to_numeric, args=({"errors":"coerce"}))
            metric_dict_pd[metric_metadata]['ds'] = pandas.to_datetime(metric_dict_pd[metric_metadata]['ds'], unit='s')
            pass
        else:
            temp_df = pandas.DataFrame(row['values'], columns=['ds', 'y']).apply(pandas.to_numeric, args=({"errors":"coerce"}))
            temp_df['ds'] = pandas.to_datetime(temp_df['ds'], unit='s')
            # print(temp_df.head())
            # print("Row Values: ",row['values']
            # print("Temp Head Before 5: \n",temp_df.head(5))
            # print("Head Before 5: \n",metric_dict_pd[metric_metadata].head(5))
            # print("Tail Before 5: \n",metric_dict_pd[metric_metadata].tail(5))
            metric_dict_pd[metric_metadata] = metric_dict_pd[metric_metadata].append(temp_df, ignore_index=True)
            # print("Head 5: \n",metric_dict_pd[metric_metadata].head(5))
            # print("Tail 5: \n",metric_dict_pd[metric_metadata].tail(5))
            mask = (metric_dict_pd[metric_metadata]['ds'] > earliest_data_time)
            metric_dict_pd[metric_metadata] = metric_dict_pd[metric_metadata].loc[mask]
            # del temp_df
            pass
        metric_dict_pd[metric_metadata] = metric_dict_pd[metric_metadata].dropna()
        metric_dict_pd[metric_metadata] = metric_dict_pd[metric_metadata].drop_duplicates('ds').sort_values(by=['ds']).reset_index(drop = True)

        if len(metric_dict_pd[metric_metadata]) == 0:
            del metric_dict_pd[metric_metadata]
            pass
        pass

        # print(metric_dict_pd[metric_metadata])
        # mask = (metric_dict_pd[metric_metadata]['ds'] > earliest_data_time) & (metric_dict_pd[metric_metadata]['ds'] <= current_time)
        # metric_dict_pd[metric_metadata] = metric_dict_pd[metric_metadata].loc[mask]
        # break
    return metric_dict_pd


def get_df_from_single_value_json(metric, metric_dict_pd={}, data_window=5):
    '''
    Method to convert a json object of a Prometheus metric to a dictionary of shaped Pandas DataFrames

    The shape is dict[metric_metadata] = Pandas Object

    Pandas Object = timestamp, value
                    15737933, 1
                    .....

    This method can also be used to update an existing dictionary with new data
    '''
    # metric_dict = {}
    current_time = datetime.now()
    earliest_data_time = current_time - timedelta(days = data_window)


    print("Pre-processing Data...........")
    # metric_dict_pd = {}
    # print("Length of metric: ", len(metric))
    for row in metric:
        # metric_dict[str(row['metric'])] = metric_dict.get(str(row['metric']),[]) + (row['values'])
        metric_metadata = str(SortedDict(row['metric']))[11:-1] # Sort the dictionary and then convert it to string so it can be hashed
        # print(metric_metadata)
        # print("Row Values: ",row['values'])
        if  metric_metadata not in metric_dict_pd:
            metric_dict_pd[metric_metadata] = pandas.DataFrame([row['value']], columns=['ds', 'y']).apply(pandas.to_numeric, args=({"errors":"coerce"}))
            metric_dict_pd[metric_metadata]['ds'] = pandas.to_datetime(metric_dict_pd[metric_metadata]['ds'], unit='s')
            pass
        else:
            temp_df = pandas.DataFrame([row['value']], columns=['ds', 'y']).apply(pandas.to_numeric, args=({"errors":"coerce"}))
            temp_df['ds'] = pandas.to_datetime(temp_df['ds'], unit='s')
            # print(temp_df.head())
            # print("Row Values: ",row['values']
            # print("Temp Head Before 5: \n",temp_df.head(5))
            # print("Head Before 5: \n",metric_dict_pd[metric_metadata].head(5))
            # print("Tail Before 5: \n",metric_dict_pd[metric_metadata].tail(5))
            metric_dict_pd[metric_metadata] = metric_dict_pd[metric_metadata].append(temp_df, ignore_index=True)
            # print("Head 5: \n",metric_dict_pd[metric_metadata].head(5))
            # print("Tail 5: \n",metric_dict_pd[metric_metadata].tail(5))
            mask = (metric_dict_pd[metric_metadata]['ds'] > earliest_data_time)
            metric_dict_pd[metric_metadata] = metric_dict_pd[metric_metadata].loc[mask]
            # del temp_df
            pass
        metric_dict_pd[metric_metadata] = metric_dict_pd[metric_metadata].dropna()
        metric_dict_pd[metric_metadata] = metric_dict_pd[metric_metadata].drop_duplicates('ds').sort_values(by=['ds']).reset_index(drop = True)

        if len(metric_dict_pd[metric_metadata]) == 0:
            del metric_dict_pd[metric_metadata]
            pass
        pass

        # print(metric_dict_pd[metric_metadata])
        # mask = (metric_dict_pd[metric_metadata]['ds'] > earliest_data_time) & (metric_dict_pd[metric_metadata]['ds'] <= current_time)
        # metric_dict_pd[metric_metadata] = metric_dict_pd[metric_metadata].loc[mask]
        # break
    return metric_dict_pd

def predict_metrics(pd_dict, prediction_range=1440):
    '''
    This Function takes input a dictionary of Pandas DataFrames, trains the Prophet model for each dataframe and returns a dictionary of predictions.
    '''

    total_label_num = len(pd_dict)
    # LABEL_LIMIT = limit_labels
    PREDICT_DURATION = prediction_range

    current_label_num = 0
    limit_iterator_num = 0

    predictions_dict = {}

    for meta_data in pd_dict:
        try:
            current_label_num += 1
            limit_iterator_num += 1

            print("Training Label {}/{}".format(current_label_num,total_label_num))
            data = pd_dict[meta_data]

            print("----------------------------------\n")
            print(meta_data)
            print("Number of Data Points: {}".format(len(pd_dict[meta_data])))
            print("----------------------------------\n")

            data['ds'] = pandas.to_datetime(data['ds'], unit='s')

            train_frame = data

            # Prophet Modelling begins here
            m = Prophet(daily_seasonality = True, weekly_seasonality=True)

            print("Fitting the train_frame")
            m.fit(train_frame)

            future = m.make_future_dataframe(periods=int(PREDICT_DURATION),freq="1MIN")

            forecast = m.predict(future)

            # To Plot
            # fig1 = m.plot(forecast)
            #
            # fig2 = m.plot_components(forecast)
            forecast['timestamp'] = forecast['ds']
            forecast = forecast[['timestamp','yhat','yhat_lower','yhat_upper']]
            forecast = forecast.set_index('timestamp')

            # Store predictions in output dictionary
            predictions_dict[meta_data] = forecast

            # forecast.plot()
            # plt.legend()
            # plt.show()
        except ValueError as exception:
            if str(exception) == "ValueError: Dataframe has less than 2 non-NaN rows.":
                print("Too many NaN values........Skipping this label")
                limit_iterator_num -= 1
            else:
                raise exception
        pass

    return predictions_dict

def fourierExtrapolation(x, n_predict, n_harm):
    n = x.size
    #n_harm = 100                     # number of harmonics in model
    t = np.arange(0, n)
    p = np.polyfit(t, x, 1)         # find linear trend in x
    x_notrend = x - p[0] * t        # detrended x
    x_freqdom = fft.fft(x_notrend)  # detrended x in frequency domain
    f = fft.fftfreq(n)              # frequencies
    indexes = np.arange(n).tolist()
    # sort indexes by frequency, lower -> higher
    indexes.sort(key = lambda i:np.absolute(f[i]))

    t = np.arange(0, n + n_predict)
    restored_sig = np.zeros(t.size)
    for i in indexes[:1 + n_harm * 2]:
        ampli = np.absolute(x_freqdom[i]) / n   # amplitude
        phase = np.angle(x_freqdom[i])          # phase
        restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
    return restored_sig + p[0] * t

def predict_metrics_fourier(pd_dict, prediction_range=1440):
    total_label_num = len(pd_dict)
    PREDICT_DURATION = prediction_range

    current_label_num = 0
    limit_iterator_num = 0

    predictions_dict = {}

    for meta_data in pd_dict:
        try:
            data = pd_dict[meta_data]
            data['ds'] = pandas.to_datetime(data['ds'], unit='s')
            vals = np.array(data["y"].tolist())

            # run model and trim forecast to only newest values
            print("Training Model......")
            forecast_vals = fourierExtrapolation(vals, prediction_range, int(len(vals)/3))
            dataframe_cols = {}
            dataframe_cols["yhat"] = np.array(forecast_vals)

            # find most recent timestamp from original data and extrapolate new
            # timestamps
            print("Creating Dummy Timestamps.....")
            min_time = min(data["ds"])
            dataframe_cols["timestamp"] = pandas.date_range(min_time, periods=len(forecast_vals), freq='min')

            # create dummy upper and lower bounds
            print("Computing Bounds....")
            upper_bound = np.mean(forecast_vals) + np.std(forecast_vals)
            lower_bound = np.mean(forecast_vals) - np.std(forecast_vals)
            dataframe_cols["yhat_upper"] = np.full((len(forecast_vals)), upper_bound)
            dataframe_cols["yhat_lower"] = np.full((len(forecast_vals)), lower_bound)

            # create series and index into precictions_dict
            print("Formatting Forecast to Pandas....")
            forecast = pandas.DataFrame(data=dataframe_cols)
            forecast = forecast.set_index('timestamp')
            predictions_dict[meta_data] = forecast

            current_label_num += 1
            limit_iterator_num += 1
        except ValueError as exception:
            if str(exception) == "ValueError: Dataframe has less than 2 non-NaN rows.":
                print("Too many NaN values........Skipping this label")
                limit_iterator_num -= 1
            else:
                raise exception
        pass

    return predictions_dict

class Accumulator:
    def __init__(self,thresh):
        self._counter = 0
        self.thresh = thresh
    def inc(self, val):
        self._counter += val
    def count(self):
        return self._counter

def detect_anomalies(predictions, data):
    if len(predictions) != len(data) :
        raise IndexError

    # parameters
    lower_bound_thresh = predictions["yhat_lower"].min()
    upper_bound_thresh = predictions["yhat_upper"].max()
    diff_thresh = 3*data["y"].std()
    acc_thresh = int(0.1*np.shape(predictions)[0])
    epsilon = .01

    diffs = []
    acc = Accumulator(acc_thresh)
    preds = np.array(predictions["yhat"])
    dat = np.array(data["y"])
    for i in range(0, np.shape(predictions)[0]):
        diff = preds[i] - dat[i]
        if abs(diff) > diff_thresh:
            # upper bound anomaly, increment counter
            acc.inc(1)
        elif dat[i] < lower_bound_thresh:
            # found trough, decrement so that acc will decay to 0
            acc.inc(-3)
        elif dat[i] > upper_bound_thresh:
            # found peak, decrement so that acc will decay to 0
            acc.inc(-3)
        else:
            # no anomaly, decrement by 2
            acc.inc(-2)

        diffs.append(max(diff, 0))

    if acc.count() > acc.thresh:
        acc_anomaly = True
    else:
        acc_anomaly = False
    w_size = int(0.8*len(data))
    w_prime_size = len(data) - w_size

    w = diffs[0:w_size]
    w_prime = diffs[w_size:]

    w_mu = np.mean(w)
    w_std = np.std(w)
    w_prime_mu = np.mean(w_prime)

    if w_std == 0:
        L_t = 0
    else:
        L_t = 1 - norm.sf((w_prime_mu - w_mu)/w_std)

    print(L_t)
    if L_t >= 1 - epsilon:
        tail_prob_anomaly = True
    else:
        tail_prob_anomaly = False

    return acc_anomaly and tail_prob_anomaly


if __name__ == "__main__":

    url = os.getenv('URL')
    token = os.getenv('BEARER_TOKEN')

    # Specific metric to run the model on
    metric_name = os.getenv('METRIC_NAME','kubelet_docker_operations_latency_microseconds')

    print("Using Metric {}.".format(metric_name))

    # This is where the model dictionary will be stored and retrieved from
    model_storage_path = "Models" + "/" + url[8:] + "/"+ metric_name + "/" + "prophet_model" + ".pkl"

    # Chunk size, download the complete data, but in smaller chunks, should be less than or equal to DATA_SIZE
    chunk_size = str(os.getenv('CHUNK_SIZE','1d'))

    # Net data size to scrape from prometheus
    data_size = str(os.getenv('DATA_SIZE','1d'))

    # Number of minutes, the model should predict the values for
    # PREDICT_DURATION=1440 # minutes, 1440 = 24 Hours

    # Limit to first few labels of the metric
    # LABEL_LIMIT = None

    # Preparing a connection to Prometheus host
    prom = Prometheus(url=url, token=token, data_chunk=chunk_size, stored_data=data_size)



    # Get metric data from Prometheus
    metric = prom.get_metric(metric_name)
    print("metric collected.")
    del prom

    # Convert data to json
    metric = json.loads(metric)

    # print(metric)

    # Metric Json is converted to a shaped dataframe
    pd_dict = get_df_from_json(metric) # This dictionary contains all the sub-labels as keys and their data as Pandas DataFrames
    del metric

    predictions = predict_metrics(pd_dict)
    for x in predictions:
        print(predictions[x].head())
    pass
