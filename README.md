<!-- # train-prometheus
A simple application to collect data from a prometheus host and train a model on it -->

# Train Prometheus
This python application has been written to deploy a training pipeline on OpenShift. This pipeline will at regular specified intervals collect new data directly from a prometheus instance and train a model on it regularly. This application also hosts a web page which can be used as a target for prometheus. This target currently serves 6 different metrics using two different prediction models (Prophet and Fourier Extrapolation).

## Getting Started

### Installing prerequisites

To run this application you will need to install several libraries listed in the requirements.txt.

To install all the dependencies at once, run the following command when inside the directory:
```
pip install -r requirements.txt
```
After all the prerequisites have been installed, open the Makefile and you will see a list of required and optional variables in the beginning.
The required variables will be used to communicate with the Prometheus and Storage end-points.

Populating the Makefile is the most important step, as you can use this to run the application on OpenShift, Docker or your local machine.

### Running on a local machine

After setting up the credentials in your Makefile, to test if Prometheus credentials are correct, run the following command:
```
make run_list_metrics
```
This will list all the metrics that are stored on the Prometheus host.

Next, to backup the previous day's metrics data to a long term block storage, run the following command:
```
make run_backup_all_metrics
```
## Running on Docker
After populating all the required variables, set the name for your docker app by changing the docker_app_name variable. Then run the following command to build the docker image.
```
make docker_build
```
This command uses the Dockerfile included in the repository to build an image. So you can use it to customize how the image is built.

Run the following command to test if the docker image is functional:
```
make docker_test
```
Your output should be something like below:
```
usage: app.py [-h] [--day DAY] [--url URL] [--token TOKEN] [--backup-all]
              [--list-metrics] [--chunk-size CHUNK_SIZE]
              [metric [metric ...]]

Backup Prometheus metrics

positional arguments:
  metric                Name of the metric, e.g. ALERTS - or --backup-all

optional arguments:
  -h, --help            show this help message and exit
  --day DAY             the day to backup in YYYYMMDD (defaults to previous
                        day)
  --url URL             URL of the prometheus server default:
                        https://prometheus-openshift-devops-monitor.1b7d.free-
                        stg.openshiftapps.com
  --token TOKEN         Bearer token for prometheus
  --backup-all          Backup all metrics
  --list-metrics        List metrics from prometheus
  --chunk-size CHUNK_SIZE
                        Size of the chunk downloaded at an instance. Accepted
                        values are 1m, 1h, 1d default: 1h

```

## Deploying on OpenShift

* ### Deploying a flask application to predict and serve the predicted metrics:
  In the Makefile set up the required variables, and then run the following command:
```
make oc_deploy
```
This will create a deployment on OpenShift and which after training the prophet model, will serve the predicted metrics as a web page (using the flask web server), these predicted metrics can later be easily collected by a prometheus instance.

Following is a sample web page view of what the metrics will look like:
```
# HELP process_virtual_memory_bytes Virtual memory size in bytes. # TYPE process_virtual_memory_bytes gauge process_virtual_memory_bytes 2834034688.0 # HELP process_resident_memory_bytes Resident memory size in bytes. # TYPE process_resident_memory_bytes gauge process_resident_memory_bytes 1400197120.0 # HELP process_start_time_seconds Start time of the process since unix epoch in seconds. # TYPE process_start_time_seconds gauge process_start_time_seconds 1534190296.29 # HELP process_cpu_seconds_total Total user and system CPU time spent in seconds. # TYPE process_cpu_seconds_total counter process_cpu_seconds_total 3421.92 # HELP process_open_fds Number of open file descriptors. # TYPE process_open_fds gauge process_open_fds 6.0 # HELP process_max_fds Maximum number of open file descriptors. # TYPE process_max_fds gauge process_max_fds 1048576.0 # HELP python_info Python platform information # TYPE python_info gauge python_info{implementation="CPython",major="3",minor="6",patchlevel="3",version="3.6.3"} 1.0 # HELP predicted_values_prophet Forecasted value from Prophet model # TYPE predicted_values_prophet gauge predicted_values_prophet{beta_kubernetes_io_arch="amd64",beta_kubernetes_io_os="linux",instance="cpt-0001.datahub.prod.upshift.rdu2.redhat.com",job="kubernetes-nodes",kubernetes_io_hostname="cpt-0001.datahub.prod.upshift.rdu2.redhat.com",node_role_kubernetes_io_compute="true",operation_type="create_container",provider="rhos",quantile="0.5",region="compute",size="small"} 33641.95330427749 # HELP predicted_values_prophet_yhat_upper Forecasted value upper bound from Prophet model # TYPE predicted_values_prophet_yhat_upper gauge predicted_values_prophet_yhat_upper{beta_kubernetes_io_arch="amd64",beta_kubernetes_io_os="linux",instance="cpt-0001.datahub.prod.upshift.rdu2.redhat.com",job="kubernetes-nodes",kubernetes_io_hostname="cpt-0001.datahub.prod.upshift.rdu2.redhat.com",node_role_kubernetes_io_compute="true",operation_type="create_container",provider="rhos",quantile="0.5",region="compute",size="small"} 39693.72640184718 # HELP predicted_values_prophet_yhat_lower Forecasted value lower bound from Prophet model # TYPE predicted_values_prophet_yhat_lower gauge predicted_values_prophet_yhat_lower{beta_kubernetes_io_arch="amd64",beta_kubernetes_io_os="linux",instance="cpt-0001.datahub.prod.upshift.rdu2.redhat.com",job="kubernetes-nodes",kubernetes_io_hostname="cpt-0001.datahub.prod.upshift.rdu2.redhat.com",node_role_kubernetes_io_compute="true",operation_type="create_container",provider="rhos",quantile="0.5",region="compute",size="small"} 27893.994518936575 # HELP predicted_values_fourier Forecasted value from Fourier Transform model # TYPE predicted_values_fourier gauge predicted_values_fourier{beta_kubernetes_io_arch="amd64",beta_kubernetes_io_os="linux",instance="cpt-0001.datahub.prod.upshift.rdu2.redhat.com",job="kubernetes-nodes",kubernetes_io_hostname="cpt-0001.datahub.prod.upshift.rdu2.redhat.com",node_role_kubernetes_io_compute="true",operation_type="create_container",provider="rhos",quantile="0.5",region="compute",size="small"} 30894.064782203226 # HELP predicted_values_fourier_yhat_upper Forecasted value upper bound from Fourier Transform model # TYPE predicted_values_fourier_yhat_upper gauge predicted_values_fourier_yhat_upper{beta_kubernetes_io_arch="amd64",beta_kubernetes_io_os="linux",instance="cpt-0001.datahub.prod.upshift.rdu2.redhat.com",job="kubernetes-nodes",kubernetes_io_hostname="cpt-0001.datahub.prod.upshift.rdu2.redhat.com",node_role_kubernetes_io_compute="true",operation_type="create_container",provider="rhos",quantile="0.5",region="compute",size="small"} 38670.03343554405 # HELP predicted_values_fourier_yhat_lower Forecasted value lower bound from Fourier Transform model # TYPE predicted_values_fourier_yhat_lower gauge predicted_values_fourier_yhat_lower{beta_kubernetes_io_arch="amd64",beta_kubernetes_io_os="linux",instance="cpt-0001.datahub.prod.upshift.rdu2.redhat.com",job="kubernetes-nodes",kubernetes_io_hostname="cpt-0001.datahub.prod.upshift.rdu2.redhat.com",node_role_kubernetes_io_compute="true",operation_type="create_container",provider="rhos",quantile="0.5",region="compute",size="small"} 27308.08660866449 # HELP http_requests_total Total HTTP Requests (count) # TYPE http_requests_total counter http_requests_total{endpoint="/",method="GET",status_code="200"} 1.0 # HELP http_requests_inprogress Number of in progress HTTP requests # TYPE http_requests_inprogress gauge http_requests_inprogress 0.0 # HELP http_request_duration_seconds HTTP request latency (seconds) # TYPE http_request_duration_seconds histogram http_request_duration_seconds_bucket{le="0.005"} 1.0 http_request_duration_seconds_bucket{le="0.01"} 1.0 http_request_duration_seconds_bucket{le="0.025"} 1.0 http_request_duration_seconds_bucket{le="0.05"} 1.0 http_request_duration_seconds_bucket{le="0.075"} 1.0 http_request_duration_seconds_bucket{le="0.1"} 1.0 http_request_duration_seconds_bucket{le="0.25"} 1.0 http_request_duration_seconds_bucket{le="0.5"} 1.0 http_request_duration_seconds_bucket{le="0.75"} 1.0 http_request_duration_seconds_bucket{le="1.0"} 1.0 http_request_duration_seconds_bucket{le="2.5"} 1.0 http_request_duration_seconds_bucket{le="5.0"} 1.0 http_request_duration_seconds_bucket{le="7.5"} 1.0 http_request_duration_seconds_bucket{le="10.0"} 1.0 http_request_duration_seconds_bucket{le="+Inf"} 1.0 http_request_duration_seconds_count 1.0 http_request_duration_seconds_sum 8.503720164299011e-05 # HELP packages_newly_added Packages newly added # TYPE packages_newly_added gauge packages_newly_added 0.0
```

## Built With

* [Requests](http://docs.python-requests.org/en/master/) - HTTP Library for python
* [Boto3](https://boto3.readthedocs.io/en/latest/reference/core/session.html) - AWS sdk for python
