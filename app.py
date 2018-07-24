from lib.prometheus import Prometheus
import pandas
import json
import time
# from lib.model import *
from lib.ceph import CephConnect as cp

from fbprophet import Prophet
import os
import gc
import pickle

# Plotting
import matplotlib.pyplot as plt

def get_df_from_json(metric):
    '''
    Method to convert a json object of a Prometheus metric to a dictionary of shaped Pandas DataFrames

    The shape is dict[metric_metadata] = Pandas Object

    Pandas Object = timestamp, value
                    15737933, 1
                    .....
    '''
    # metric_dict = {}
    print("Shaping Data...........")
    metric_dict_pd = {}
    # print("Length of metric: ", len(metric))
    for row in metric:
        # metric_dict[str(row['metric'])] = metric_dict.get(str(row['metric']),[]) + (row['values'])
        metric_metadata = str(row['metric'])
        # print(metric_metadata)
        # print("Row Values: ",row['values'])
        if  metric_metadata not in metric_dict_pd:
            metric_dict_pd[metric_metadata] = pandas.DataFrame(row['values'], columns=['ds', 'y']).apply(pandas.to_numeric, args=({"errors":"coerce"}))
            pass
        else:
            temp_df = pandas.DataFrame(row['values'], columns=['ds', 'y']).apply(pandas.to_numeric, args=({"errors":"coerce"}))
            # print(temp_df.head())
            # print("Row Values: ",row['values']
            metric_dict_pd[metric_metadata] = pandas.concat([metric_dict_pd[metric_metadata], temp_df])
            # del temp_df
            pass
        pass
        metric_dict_pd[metric_metadata].set_index('ds')
        # break
    return metric_dict_pd


url = os.getenv('URL')
token = os.getenv('BEARER_TOKEN')

# Chunk size, download the complete data, but in smaller chunks, should be less than or equal to DATA_SIZE
chunk_size = str(os.getenv('CHUNK_SIZE','3h'))

# Net data size to scrape from prometheus
data_size = str(os.getenv('DATA_SIZE','3h'))

# Number of minutes, the model should predict the values for
PREDICT_DURATION=1440

# Preparing a connection to Prometheus host
prom = Prometheus(url=url, token=token, data_chunk=chunk_size, stored_data=data_size)

# Specific metric to run the model on
metric_name = 'kubelet_docker_operations_latency_microseconds'
print("Using Metric {}.".format(metric_name))

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

model_dict = {} # Dictionary where all the models will be stored

current_label_num = 0
total_label_num = len(pd_dict)
print("Numer of labels: {} \n".format(total_label_num))

for meta_data in pd_dict:
    try:
        current_label_num+=1
        print("Training Label {}/{}".format(current_label_num,total_label_num))
        data = pd_dict[meta_data]

    # print(meta_data, data)

    # print(len(new_dict))
    # for key in new_dict:

        print("----------------------------------\n")
        print(meta_data)
        print("----------------------------------\n")
    # print(new_dict[key])
    # data_pd = new_dict[key]



        data['ds'] = pandas.to_datetime(data['ds'], unit='s')
    # data = data.set_index('ds')
    # print(data.info())
    # print(data.head())
    # print(data.index)
    # data = data.cumsum()
    # data.plot()
    # plt.figure()

    # To PLot:

    # data.plot(style='k--', label='Series')
    # plt.legend()
    # plt.show()
        train_frame = data
        # train_frame = data[0 : int(0.7*len(data))]
        # test_frame = data[int(0.7*len(data)) : ]

        print(len(train_frame))
        print(train_frame.head())
        # print(test_frame.head())
    # train_frame['y'] = train_frame['value']
    # train_frame['ds'] = train_frame['timestamp']

    # time.sleep(1)
        model_dict[meta_data] = Prophet()
        print("Fitting the train_frame")
        model_dict[meta_data].fit(train_frame)
    #
        future = model_dict[meta_data].make_future_dataframe(periods=int(PREDICT_DURATION),freq="1MIN")
        # future = m.make_future_dataframe(periods=int(len(test_frame) * 1.1),freq="1MIN")
        forecast = model_dict[meta_data].predict(future)
    # print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

        # fig1 = m.plot(forecast)
        #
        # fig2 = m.plot_components(forecast)
    # future = m.make_future_dataframe(periods= int(len(test_frame) * 1.1),freq= '1MIN')
    # print("----------------------------------\n")
    # gc.collect()
    # print("Predicting Future.................")
    # forecast = m.predict(future)
    # print(forecast.head())
    #
    # forecasted_features = ['ds','yhat','yhat_lower','yhat_upper']
    # # m.plot(forecast,xlabel="Time stamp",ylabel="Value");
    # # m.plot_components(forecast);
    #
    # forecast = forecast[forecasted_features]
    # print(forecast.head())
    #

        # fig1 = model_dict[meta_data].plot(forecast)
        #
        # fig2 = model_dict[meta_data].plot_components(forecast)
        forecast['timestamp'] = forecast['ds']
        forecast['values'] = data['y']
    # forecast.plot()
        forecast = forecast[['timestamp','values','yhat','yhat_lower','yhat_upper']]
        forecast = forecast.set_index('timestamp')

        # forecast.plot()
        # plt.legend()
        # plt.show()
    except ValueError:
        print("Too many NaN values")
    pass


# output['values'] = forecast[['timestamp','yhat']].to_json()
# output_json = json.dumps(output)

#
file_name = 'prophet_model.pkl'
file = open(file_name, 'wb')
pickle.dump(model_dict, file)
file.close()
# Store Forecast to CEPH

session = cp()
object_path = "Models" + "/" + url[8:] + "/"+ metric_name + "/" + "prophet_model" + ".pkl"
print(session.store_data(name = metric_name,
                        object_path = object_path,
                        values = pickle.dumps(model_dict)))




# forecast.to_json()
# break


#     print(len(metric_dict[key]))
#     break
#     pass
# test_df.set_index('timestamp')
# test_df['timestamp'] = pandas.to_datetime(test_df['timestamp'], unit='s')
# print(test_df.head(100))

# pd_metric = pandas.DataFrame.from_dict(metric_dict)

# print(pd_metric.info)
# for x in metric:
#     print((x['metric']).sort())
#     pass
# print(metric[0])
# print("----------------------------------\n")
# pd_metric = pandas.read_json(metric)
# print(pd_metric.info())
# # for x in (pd_metric['values']):
#     # print(x)
#     # pass
