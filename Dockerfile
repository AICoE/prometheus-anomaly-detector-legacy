FROM docker.io/centos/python-36-centos7:latest


ADD requirements.txt /
RUN pip install -r /requirements.txt

ADD app.py /
ADD prometheus.py /
ADD model.py /
ADD ceph.py /
ADD lib /lib
 

CMD [ "python", "/app.py"]
