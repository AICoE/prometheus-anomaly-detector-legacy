import boto3
import bz2
import os
import pickle
import botocore

class CephConnect:
    def __init__(self, access_key = None, secret_key = None, object_store = None, object_store_endpoint = None):
      self.boto_settings = {
          'access_key': os.getenv('BOTO_ACCESS_KEY', access_key),
          'secret_key': os.getenv('BOTO_SECRET_KEY', secret_key),
          'object_store': os.getenv('BOTO_OBJECT_STORE', object_store),
          'object_store_endpoint': os.getenv('BOTO_STORE_ENDPOINT', object_store_endpoint)
      }

    def store_data(self, name, values, object_path = None):
        '''
        Function to store predictions to ceph
        '''
        if not values:
            return "No values for {}".format(name)
        # Create a session with CEPH (or any black storage) storage with the stored credentials
        session = boto3.Session(
            aws_access_key_id=self.boto_settings['access_key'],
            aws_secret_access_key=self.boto_settings['secret_key']
        )

        s3 = session.resource('s3',
                              endpoint_url=self.boto_settings['object_store_endpoint'],
                              verify=False)
        # prometheus-openshift-devops-monitor.a3c1.starter-us-west-1.openshiftapps.com/container_cpu_usage_percent_by_host/201807040259.json.bz2
        if not object_path:
            object_path = str(name)
            pass
        object_path = object_path + ".bz2"
        try:
            payload = bz2.compress(values.encode('utf-8'))

        except AttributeError:
            payload = bz2.compress(values)
        rv = s3.meta.client.put_object(Body=payload,
                                       Bucket=self.boto_settings['object_store'],
                                       Key=object_path)
        if rv['ResponseMetadata']['HTTPStatusCode'] == 200:
            return object_path
        else:
            return str(rv)

    def get_model_dict(self, model_storage_path):
        session = boto3.Session(
            aws_access_key_id=self.boto_settings['access_key'],
            aws_secret_access_key=self.boto_settings['secret_key']
        )

        s3 = session.resource('s3',
                              endpoint_url=self.boto_settings['object_store_endpoint'],
                              verify=False)
        # try to get model from ceph
        try:
            model_storage_path = model_storage_path + ".bz2"
            print("receiveing Object from: \n {}".format(model_storage_path))

            received_object = s3.Object(self.boto_settings['object_store'], model_storage_path).get()['Body'].read()
            # print(type(received_object))
            model_dict = pickle.loads(bz2.decompress(received_object))
            # print(model_dict.keys())
        except botocore.exceptions.ClientError as exc:
            if exc.response['Error']['Code'] in ('404', 'NoSuchKey'):
                # if no model in ceph, return an empty model dictionary
                print("Stored Model not found")
                model_dict = {}
        return model_dict
