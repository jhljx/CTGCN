FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-runtime

MAINTAINER jhljx8918@gmail.com

RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pandas==1.0.5 networkx==2.4 matplotlib==3.3.0 scipy==1.5.1 scikit-learn==0.23.1 torch-geometric==1.6.0

RUN apt-get update && apt-get install -y wget
RUN wget -P /tmp -c https://pytorch-geometric.com/whl/torch-1.5.0/torch_scatter-2.0.5%2Bcu101-cp37-cp37m-linux_x86_64.whl && pip install /tmp/torch_scatter-2.0.5+cu101-cp37-cp37m-linux_x86_64.whl
RUN wget -P /tmp -c https://pytorch-geometric.com/whl/torch-1.5.0/torch_cluster-1.5.6%2Bcu101-cp37-cp37m-linux_x86_64.whl && pip install /tmp/torch_cluster-1.5.6+cu101-cp37-cp37m-linux_x86_64.whl
RUN wget -P /tmp -c https://pytorch-geometric.com/whl/torch-1.5.0/torch_sparse-0.6.6%2Bcu101-cp37-cp37m-linux_x86_64.whl && pip install /tmp/torch_sparse-0.6.6+cu101-cp37-cp37m-linux_x86_64.whl
RUN wget -P /tmp -c https://pytorch-geometric.com/whl/torch-1.5.0/torch_spline_conv-1.2.0%2Bcu101-cp37-cp37m-linux_x86_64.whl && pip install /tmp/torch_spline_conv-1.2.0+cu101-cp37-cp37m-linux_x86_64.whl
