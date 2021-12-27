# ARG FROM_IMAGE=ubuntu:18.04
ARG FROM_IMAGE=nvidia/cuda:10.2-cudnn8-devel-ubuntu18.04

FROM ${FROM_IMAGE}

RUN apt update && apt install ca-certificates -y

# # change tsinghua mirror
# RUN echo \
# "deb [trusted=yes] https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic main restricted universe multiverse\n\
# deb [trusted=yes] https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-updates main restricted universe multiverse\n\
# deb [trusted=yes] https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-backports main restricted universe multiverse\n\
# deb [trusted=yes] https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-security main restricted universe multiverse" > /etc/apt/sources.list

RUN apt update && apt install wget \
        python3.7 python3.7-dev \
        g++ build-essential openssh-server -y

WORKDIR /usr/src/jittor

RUN apt download python3-distutils && dpkg-deb -x ./python3-distutils* / \
    && wget -O - https://bootstrap.pypa.io/get-pip.py | python3.7

ENV PYTHONIOENCODING utf8

# # change tsinghua mirror
# RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip3 install  \
        numpy \
        tqdm \
        pillow \
        astunparse \
        notebook

RUN pip3 install matplotlib

RUN apt install openmpi-bin openmpi-common libopenmpi-dev -y

RUN pip3 install jittor --timeout 100 && python3.7 -m jittor.test.test_example

# RUN apt install git -y

# RUN git clone https://github.com/Jittor/jittor.git /opt/ml/code/jittor

# WORKDIR /opt/ml/code/jittor

# RUN pip3 uninstall jittor -y

# COPY . .

# RUN pip3 install . --timeout 100

# RUN python3.7 -m jittor.test.test_example

# CMD python3.7 -m jittor.notebook --allow-root --ip=0.0.0.0

# Install nginx notebook
RUN apt-get install -y --no-install-recommends \
         wget \
         nginx \
         ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# forward request and error logs to docker log collector
RUN ln -sf /dev/stdout /var/log/nginx/access.log
RUN ln -sf /dev/stderr /var/log/nginx/error.log

RUN pip3 install flask gevent gunicorn boto3

ENV PATH="/opt/ml/code:${PATH}"

# /opt/ml and all subdirectories are utilized by SageMaker, we use the /code subdirectory to store our user code.
RUN mkdir -p /opt/ml/code
COPY train /opt/ml/code
COPY train.py /opt/ml/code
COPY serve /opt/ml/code
COPY wsgi.py /opt/ml/code
COPY predictor.py /opt/ml/code
COPY nginx.conf /opt/ml/code

WORKDIR /opt/ml/code