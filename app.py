from lib.prometheus import Prometheus
import pandas
import json
import time
# from lib.model import *
from lib.ceph import CephConnect as cp

from fbprophet import Prophet
import os
import gc


# def get_df_from_json(metric):
#     print("Here")
#     '''
#     Method to convert a json object of a Prometheus metric to a dictionary of shaped Pandas DataFrames
#
#     The shape is dict[metric_metadata] = Pandas Object
#
#     Pandas Object = timestamp, value
#                     15737933, 1
#                     .....
#     '''
#     # metric_dict = {}
#     print("Shaping Data...........")
#     metric_dict_pd = {}
#     for row in metric:
#         # metric_dict[str(row['metric'])] = metric_dict.get(str(row['metric']),[]) + (row['values'])
#         print("Row Values: ",row['values'])
#         metric_metadata = str(row['metric'])
#         if  metric_metadata not in metric_dict_pd:
#             metric_dict_pd[metric_metadata] = pandas.DataFrame(columns=['timestamp', 'value'])
#             pass
#         else:
#             temp_df = pandas.DataFrame(row['values'], columns=['timestamp', 'value'])
#             # print("Row Values: ",row['values']
#             metric_dict_pd[metric_metadata] = pandas.concat([metric_dict_pd[metric_metadata], temp_df])
#             # del temp_df
#             pass
#         pass
#         # metric_dict_pd[metric_metadata].set_index('timestamp')
#     return metric_dict_pd

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
        print("Row Values: ",row['values'])
        if  metric_metadata not in metric_dict_pd:
            metric_dict_pd[metric_metadata] = pandas.DataFrame(row['values'], columns=['ds', 'y'])
            pass
        else:
            temp_df = pandas.DataFrame(row['values'], columns=['ds', 'y'])
            print(temp_df.head())
            # print("Row Values: ",row['values']
            metric_dict_pd[metric_metadata] = pandas.concat([metric_dict_pd[metric_metadata], temp_df])
            # del temp_df
            pass
        pass
        # metric_dict_pd[metric_metadata].set_index('timestamp')
        break
    return metric_dict_pd


url = os.getenv('URL')
token = os.getenv('BEARER_TOKEN')
chunk_size = str(os.getenv('CHUNK_SIZE','5m'))
data_size = str(os.getenv('DATA_SIZE','5m'))

prom = Prometheus(url=url, token=token, data_chunk=chunk_size, stored_data=data_size)

metric_name = 'kubelet_docker_operations_latency_microseconds'

print("Using Metric {}.".format(metric_name))

metric = prom.get_metric(metric_name)

del prom

metric = json.loads(metric)

# print(metric)
print("----------------------------------\n")


# Metric Json is converted to a shaped dataframe
new_dict = get_df_from_json(metric)

del metric

for key, value in new_dict.items():
    meta_data = key
    data = value

    break
    pass
print(meta_data, data)

del new_dict
# print(len(new_dict))
# for key in new_dict:
key = meta_data
print("----------------------------------\n")
print(key)
print("----------------------------------\n")
# print(new_dict[key])
# data_pd = new_dict[key]
train_frame = data[0 : int(0.7*len(data))]
test_frame = data[int(0.7*len(data)) : ]

print(len(train_frame), len(test_frame), len(data))
print(train_frame.head())
print(test_frame.head())
print(data.head())
# train_frame['y'] = train_frame['value']
# train_frame['ds'] = train_frame['timestamp']

# time.sleep(1)
m = Prophet()
print("Fitting the train_frame")
m.fit(train_frame)

future = m.make_future_dataframe(periods= int(len(test_frame) * 1.1),freq= '1MIN')
print("----------------------------------\n")
gc.collect()
print("Predicting Future.................")
forecast = m.predict(future)
print(forecast.head())

forecasted_features = ['ds','yhat','yhat_lower','yhat_upper']
# m.plot(forecast,xlabel="Time stamp",ylabel="Value");
# m.plot_components(forecast);

forecast = forecast[forecasted_features]
print(forecast.head())

forecast['timestamp'] = forecast['ds']
forecast['values'] = data_pd['values']
forecast = forecast[['timestamp','values','yhat','yhat_lower','yhat_upper']]
output_json = key + forecast.to_json()

# Store Forecast to CEPH
session = cp()
object_path = "Predictions" + "/" + prom_host + "/" + url + "/" + "Predictions" + ".json"
print(session.store_data(name = metric_name, object_path = object_path, values = forecast.to_json()))
# file = open('Predictions.json', 'w')
# file.write(output_json)
# file.close()
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
