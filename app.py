from lib.prometheus import Prometheus
import pandas
import json
from lib.model import *
from lib.ceph import CephConnect as cp

from fbprophet import Prophet
import os

url = os.getenv('URL')
token = os.getenv('BEARER_TOKEN')

prom = Prometheus(url=url, token=token)

metric_name = prom.all_metrics()[1]

print("Using Metric {}.".format(metric_name))

metric = prom.get_metric(metric_name)


metric = json.loads(metric)

# print(metric)
# print("----------------------------------\n")


# Metric Json is converted to a shaped dataframe
new_dict = get_df_from_json(metric)
del metric

for key in new_dict:
    print("----------------------------------\n")
    print(key)
    print("----------------------------------\n")
    # print(new_dict[key])
    # data_pd = new_dict[key]
    train_frame = new_dict[key][0 : int(0.7*len(new_dict[key]))]
    test_frame = new_dict[key][int(0.7*len(new_dict[key])) : ]

    print(len(train_frame), len(test_frame), len(new_dict[key]))
    print(train_frame.head())
    print(test_frame.head())
    print(new_dict[key].head())
    train_frame['y'] = train_frame['value']
    train_frame['ds'] = train_frame['timestamp']

    m = Prophet()

    m.fit(train_frame)

    future = m.make_future_dataframe(periods= int(len(test_frame) * 1.1),freq= '1MIN')
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
    break


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
