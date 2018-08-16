import random
import time
import os
import sys
import bz2
import pandas
import argparse
import pickle
from flask import Flask, render_template_string, abort, Response
from datetime import datetime, timedelta
from prometheus_client import CollectorRegistry, generate_latest, REGISTRY, Counter, Gauge, Histogram
from prometheus import Prometheus
from model import *
from ceph import CephConnect as cp
from ast import literal_eval
# Scheduling stuff
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
import atexit
from scipy.stats import norm
import numpy as np

app = Flask(__name__)

data_window = int(os.getenv('DATA_WINDOW_SIZE',60)) # Number of days of past data, the model should use to train

url = os.getenv('URL')
token = os.getenv('BEARER_TOKEN')

# Specific metric to run the model on
metric_name = os.getenv('METRIC_NAME','kubelet_docker_operations_latency_microseconds')

print("Using Metric {}.".format(metric_name))

# This is where the model dictionary will be stored and retrieved from
data_storage_path = "Data_Frames" + "/" + url[8:] + "/"+ metric_name + "/" + "prophet_model" + ".pkl"

# Chunk size, download the complete data, but in smaller chunks, should be less than or equal to DATA_SIZE
chunk_size = str(os.getenv('CHUNK_SIZE','2d'))

# Net data size to scrape from prometheus
data_size = str(os.getenv('DATA_SIZE','2d'))

train_schedule = int(os.getenv('TRAINING_REPEAT_HOURS',6))


TRUE_LIST = ["True", "true", "1", "y"]

store_intermediate_data = os.getenv("STORE_INTERMEDIATE_DATA", "False") # Setting this to true will store intermediate dataframes to ceph


if str(os.getenv('GET_OLDER_DATA',"False")) in TRUE_LIST:
    print("Collecting previously stored data.........")
    data_dict = cp().get_latest_df_dict(data_storage_path)
    pass
else:
    data_dict = {}


default_label_config = "{'__name__': 'kubelet_docker_operations_latency_microseconds', 'beta_kubernetes_io_arch': 'amd64', 'beta_kubernetes_io_os': 'linux', 'instance': 'cpt-0001.ocp.prod.upshift.eng.rdu2.redhat.com', 'job': 'kubernetes-nodes', 'kubernetes_io_hostname': 'cpt-0001.ocp.prod.upshift.eng.rdu2.redhat.com', 'operation_type': 'version', 'provider': 'rhos', 'quantile': '0.5', 'region': 'compute', 'size': 'small'} "
default_label_config = default_label_config +";" + "{'__name__': 'kubelet_docker_operations_latency_microseconds', 'beta_kubernetes_io_arch': 'amd64', 'beta_kubernetes_io_os': 'linux', 'instance': 'cpt-0001.ocp.prod.upshift.eng.rdu2.redhat.com', 'job': 'kubernetes-nodes', 'kubernetes_io_hostname': 'cpt-0001.ocp.prod.upshift.eng.rdu2.redhat.com', 'operation_type': 'version', 'provider': 'rhos', 'quantile': '0.9', 'region': 'compute', 'size': 'small'} "

config_list = []
fixed_label_config = str(os.getenv("LABEL_CONFIG",default_label_config))
if fixed_label_config  != "None":
    config_list = fixed_label_config.split(";") # Separate multiple label configurations using a ';' (semi-colon)
    fixed_label_config_dict = literal_eval(config_list[0]) # # TODO: Add more error handling here


predictions_dict_prophet = {}
predictions_dict_fourier = {}
current_metric_metadata = ""
current_metric_metadata_dict = {}

def detect_anomalies(predictions, data):
    if len(predictions) != len(data) :
        raise IndexError
    
    # parameters
    lower_bound_thresh = predictions["yhat_lower"].min() 
    upper_bound_thresh = predictions["yhat_upper"].max() 
    diff_thresh = 2*data["y"].std() 
    acc_thresh = int(0.1*np.shape(predictions)[0])
    epsilon = .1 

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

# iteration = 0
def job(current_time):
    # TODO: Replace this function with model training function and set up the correct IntervalTrigger time
    global data_dict, predictions_dict_prophet, predictions_dict_fourier, current_metric_metadata, current_metric_metadata_dict, data_window, url, token, chunk_size, data_size, TRUE_LIST, store_intermediate_data
    global data, config_list
    # iteration += 1
    start_time = time.time()
    prom = Prometheus(url=url, token=token, data_chunk=chunk_size, stored_data=data_size)
    metric = prom.get_metric(metric_name)
    print("metric collected.")

    # Convert data to json
    metric = json.loads(metric)

    # Metric Json is converted to a shaped dataframe
    data_dict = get_df_from_json(metric, data_dict, data_window) # This dictionary contains all the sub-labels as keys and their data as Pandas DataFrames
    del metric, prom

    if str(store_intermediate_data) in TRUE_LIST:
        print("DataFrame stored at: ",cp().store_data(metric_name, pickle.dumps(data_dict), (data_storage_path + str(datetime.now().strftime('%Y%m%d%H%M')))))
        pass


    if fixed_label_config != "None": #If a label config has been specified
        single_label_data_dict = {}

        # split into multiple label configs
        existing_config_list = list(data_dict.keys())
        for config in config_list:
            config_found = False
            for existing_config in existing_config_list:
                if SortedDict(literal_eval(existing_config)) == SortedDict(literal_eval(config)):
                    single_label_data_dict[existing_config] = data_dict[existing_config]
                    config_found = True
                    pass
            if not config_found:
                print("Specified Label Configuration {} was not found".format(config))
                raise KeyError
                pass
            # single_label_data_dict[config] = data_dict[config]
            pass

        # single_label_data_dict[fixed_label_config] = data_dict[fixed_label_config]
        current_metric_metadata = list(single_label_data_dict.keys())[0]
        current_metric_metadata_dict = literal_eval(current_metric_metadata)

        print(data_dict[current_metric_metadata].head(5))
        print(data_dict[current_metric_metadata].tail(5))

        print("Using the default label config")
        predictions_dict_prophet = predict_metrics(single_label_data_dict)
        # print(single_label_data_dict)
        predictions_dict_fourier = predict_metrics_fourier(single_label_data_dict)
        pass
    else:
        for x in data_dict:
            print(data_dict[x].head(5))
            print(data_dict[x].tail(5))
            break
            pass
        predictions_dict_prophet = predict_metrics(data_dict)
        predictions_dict_fourier = predict_metrics_fourier(data_dict)

    # TODO: Trigger Data Pruning here
    function_run_time = time.time() - start_time

    print("Total time taken to train was: {} seconds.".format(function_run_time))
    pass

job(datetime.now())

# Schedular schedules a background job that needs to be run regularly
scheduler = BackgroundScheduler()
scheduler.start()
scheduler.add_job(
    func=lambda: job(datetime.now()),
    trigger=IntervalTrigger(hours=train_schedule),
    id='training_job',
    name='Train Prophet model every day regularly',
    replace_existing=True)

# Shut down the scheduler when exiting the app
atexit.register(lambda: scheduler.shutdown())



#Multiple gauges set for the predicted values
print("current_metric_metadata_dict: ", current_metric_metadata_dict)
PREDICTED_VALUES_PROPHET = Gauge('predicted_values_prophet', 'Forecasted value from Prophet model', [label for label in current_metric_metadata_dict if label != "__name__"])
PREDICTED_VALUES_PROPHET_UPPER = Gauge('predicted_values_prophet_yhat_upper', 'Forecasted value upper bound from Prophet model', [label for label in current_metric_metadata_dict if label != "__name__"])
PREDICTED_VALUES_PROPHET_LOWER = Gauge('predicted_values_prophet_yhat_lower', 'Forecasted value lower bound from Prophet model', [label for label in current_metric_metadata_dict if label != "__name__"])

PREDICTED_VALUES_FOURIER = Gauge('predicted_values_fourier', 'Forecasted value from Fourier Transform model', [label for label in current_metric_metadata_dict if label != "__name__"])
PREDICTED_VALUES_FOURIER_UPPER = Gauge('predicted_values_fourier_yhat_upper', 'Forecasted value upper bound from Fourier Transform model', [label for label in current_metric_metadata_dict if label != "__name__"])
PREDICTED_VALUES_FOURIER_LOWER = Gauge('predicted_values_fourier_yhat_lower', 'Forecasted value lower bound from Fourier Transform model', [label for label in current_metric_metadata_dict if label != "__name__"])

# Standard Flask route stuff.
@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/metrics')
def metrics():

    global predictions_dict_prophet, predictions_dict_fourier, current_metric_metadata, current_metric_metadata_dict
    for metadata in predictions_dict_prophet:

        #Find the index matching with the current timestamp
        index_prophet = predictions_dict_prophet[metadata].index.get_loc(datetime.now(), method='nearest')
        index_fourier = predictions_dict_fourier[metadata].index.get_loc(datetime.now(), method='nearest')
        current_metric_metadata = metadata

        print("The current time is: ",datetime.now())
        print("The matching index for Prophet model found was: \n", predictions_dict_prophet[metadata].iloc[[index_prophet]])
        print("The matching index for Fourier Transform found was: \n", predictions_dict_fourier[metadata].iloc[[index_fourier]])

        current_metric_metadata_dict = literal_eval(metadata)

        temp_current_metric_metadata_dict = current_metric_metadata_dict.copy()

        del temp_current_metric_metadata_dict["__name__"]

        # Update the metric values for prophet model
        PREDICTED_VALUES_PROPHET.labels(**temp_current_metric_metadata_dict).set(predictions_dict_prophet[metadata]['yhat'][index_prophet])
        PREDICTED_VALUES_PROPHET_UPPER.labels(**temp_current_metric_metadata_dict).set(predictions_dict_prophet[metadata]['yhat_upper'][index_prophet])
        PREDICTED_VALUES_PROPHET_LOWER.labels(**temp_current_metric_metadata_dict).set(predictions_dict_prophet[metadata]['yhat_lower'][index_prophet])

        # Update the metric values for fourier transform model
        PREDICTED_VALUES_FOURIER.labels(**temp_current_metric_metadata_dict).set(predictions_dict_fourier[metadata]['yhat'][index_fourier])
        PREDICTED_VALUES_FOURIER_UPPER.labels(**temp_current_metric_metadata_dict).set(predictions_dict_fourier[metadata]['yhat_upper'][index_fourier])
        PREDICTED_VALUES_FOURIER_LOWER.labels(**temp_current_metric_metadata_dict).set(predictions_dict_fourier[metadata]['yhat_lower'][index_fourier])

        pass

    return Response(generate_latest(REGISTRY).decode("utf-8"), content_type='text; charset=utf-8')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
    pass
