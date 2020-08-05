FROM ubuntu

MAINTAINER jhljx8918@gmail.com

RUN apt update && apt install python3.7 python3-pip gcc g++ --assume-yes && python3 -m pip install --upgrade pip \
    && pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy==1.18.1 pandas==1.0.5 networkx==2.4 matplotlib==3.3.0 scipy==1.5.1 scikit-learn==0.23.1 cython==0.29.21 \
    torch==1.5.1 torch-cluster==1.5.6 torch-scatter==2.0.5 torch-sparse==0.6.6 torch-spline-conv==1.2.0 torch-geometric==1.6.0