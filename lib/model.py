import pandas
import json

def get_df_from_json(metric):
    # metric_dict = {}
    metric_dict_pd = {}
    for row in metric:
        # metric_dict[str(row['metric'])] = metric_dict.get(str(row['metric']),[]) + (row['values'])
        metric_metadata = str(row['metric'])
        if  metric_metadata not in metric_dict_pd:
            metric_dict_pd[metric_metadata] = pandas.DataFrame(columns=['timestamp', 'value'])
            pass
        else:
            # for value in (row['values']):
                # print(value)
            temp_df = pandas.DataFrame(row['values'], columns=['timestamp', 'value'])
            # print(temp_df.head())
            metric_dict_pd[metric_metadata] = pandas.concat([metric_dict_pd[metric_metadata], temp_df])
            del temp_df
            pass
        pass
        metric_dict_pd[metric_metadata].set_index('timestamp')
        # print(metric_dict_pd[metric_metadata])
        # metric_dict_pd[metric_metadata]['timestamp'] = pandas.to_datetime(metric_dict_pd[metric_metadata]['timestamp'], unit='s')
    return metric_dict_pd
