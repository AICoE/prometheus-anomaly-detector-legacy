import random
import time
import os
import sys
import bz2
import pandas
import argparse
import pickle
from flask import Flask, render_template_string, abort
from datetime import datetime, timedelta
from prometheus_client import CollectorRegistry, generate_latest, REGISTRY, Counter, Gauge, Histogram
from prometheus import Prometheus
from model import *
from ceph import CephConnect as cp

# Scheduling stuff
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
import atexit

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
chunk_size = str(os.getenv('CHUNK_SIZE','1d'))

# Net data size to scrape from prometheus
data_size = str(os.getenv('DATA_SIZE','1d'))

train_schedule = int(os.getenv('TRAINING_REPEAT_HOURS',24))

data_dict = {}
predictions_dict = {}
current_metric_metadata = ""
# iteration = 0
def job(current_time):
    # TODO: Replace this function with model training function and set up the correct IntervalTrigger time
    global data_dict, predictions_dict, current_metric_metadata, data_window, url, token, chunk_size, data_size
    global data
    # iteration += 1
    prom = Prometheus(url=url, token=token, data_chunk=chunk_size, stored_data=data_size)
    metric = prom.get_metric(metric_name)
    print("metric collected.")

    # Convert data to json
    metric = json.loads(metric)

    # Metric Json is converted to a shaped dataframe
    data_dict = get_df_from_json(metric, data_dict, data_window) # This dictionary contains all the sub-labels as keys and their data as Pandas DataFrames
    del metric, prom
    print("DataFrame stored at: ",cp().store_data(metric_name, pickle.dumps(data_dict), (data_storage_path + str(datetime.now().strftime('%Y%m%d%H%M')))))
    for x in data_dict:
        # if (len(data_dict[x].dropna()) > 100):
        print(data_dict[x].head(5))
        print(data_dict[x].tail(5))
        # data_dict[x] = data_dict[x].reset_index(drop = True).sort_values(by=['ds'])
        # print(data_dict[x].head(5))
        # print(data_dict[x].tail(5))
        break
        pass

    predictions_dict = predict_metrics(data_dict)
    for key in predictions_dict:
        current_metric_metadata = key
        data = predictions_dict[key]
        break # use the first predicted metric
        # print(len(data[~data.index.duplicated()]))
        pass
    # print("Data Head: \n",data.head(5))
    # print("Data Tail: \n",data.tail(5))
    # data['timestamp'] = data['ds']

    # data = data.set_index(data['timestamp'])
    data = data[~data.index.duplicated()]
    # data = data[]
    data = data.sort_index()
    # data = data[:datetime.now()]
    # TODO: Trigger Data Pruning here
    pass

job(datetime.now())

# Schedular schedules a background job that needs to be run regularly
scheduler = BackgroundScheduler()
scheduler.start()
scheduler.add_job(
    func=lambda: job(datetime.now()),
    trigger=IntervalTrigger(hours=train_schedule),# change this to a different interval
    id='training_job',
    name='Train Prophet model every day regularly',
    replace_existing=True)
# Shut down the scheduler when exiting the app
atexit.register(lambda: scheduler.shutdown())

#Parsing the required arguments
# parser = argparse.ArgumentParser(description='Service metrics')
# parser.add_argument('--file', type=str, help='The filename of predicted values to read from', default="predictions.json")
#
# args = parser.parse_args()
# #Read the JSON file
# data = pandas.read_json(args.file)
# print(data.head())

# modify the DataFrame
# data = data.set_index(data['timestamp'])
# data = data[~data.index.duplicated()]
# # data = data[]
# data = data.sort_index()
# data = data[:datetime.now()]

#A gauge set for the predicted values
PREDICTED_VALUES = Gauge('predicted_values', 'Forecasted values from Prophet', ['value_type'])

# A counter to count the total number of HTTP requests
REQUESTS = Counter('http_requests_total', 'Total HTTP Requests (count)', ['method', 'endpoint', 'status_code'])

# A gauge (i.e. goes up and down) to monitor the total number of in progress requests
IN_PROGRESS = Gauge('http_requests_inprogress', 'Number of in progress HTTP requests')

# A histogram to measure the latency of the HTTP requests
TIMINGS = Histogram('http_request_duration_seconds', 'HTTP request latency (seconds)')

# A gauge to count the number of packages newly added
PACKAGES_NEW = Gauge('packages_newly_added', 'Packages newly added')


# Standard Flask route stuff.
@app.route('/')
# Helper annotation to measure how long a method takes and save as a histogram metric.
@TIMINGS.time()
# Helper annotation to increment a gauge when entering the method and decrementing when leaving.
@IN_PROGRESS.track_inprogress()
def hello_world():
    REQUESTS.labels(method='GET', endpoint="/", status_code=200).inc()  # Increment the counter
    return 'Hello, World!'


@app.route('/hello/<name>')
@IN_PROGRESS.track_inprogress()
@TIMINGS.time()
def index(name):
    REQUESTS.labels(method='GET', endpoint="/hello/<name>", status_code=200).inc()
    return render_template_string('<b>Hello {{name}}</b>!', name=name)

@app.route('/packages')
def countpkg():
	for i in range(10):
		packages_added = True
		if packages_added:
			PACKAGES_NEW.inc()
	return render_template_string('Counting packages....')

@app.route('/metrics')
def metrics():
    #Find the index matching with the current timestamp
    global data
    index = data.index.get_loc(datetime.now(), method='nearest')
    print("The current time is: ",datetime.now())
    print("The matching index found:", index, "nearest row item is: \n", data.iloc[[index]])
    #Set the Gauge with the predicted values of the index found
    PREDICTED_VALUES.labels(value_type=current_metric_metadata + 'yhat').set(data['yhat'][index])
    PREDICTED_VALUES.labels(value_type='yhat_upper').set(data['yhat_upper'][index])
    PREDICTED_VALUES.labels(value_type='yhat_lower').set(data['yhat_lower'][index])
    return generate_latest(REGISTRY)

@app.route('/prometheus')
@IN_PROGRESS.track_inprogress()
@TIMINGS.time()
def display():
	REQUESTS.labels(method='GET', endpoint="/metrics", status_code=200).inc()
	return generate_latest(REGISTRY)

if __name__ == "__main__":
    app.run(port=8080)
    pass
