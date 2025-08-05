FROM nvidia/cudagl:11.4.2-base-ubuntu20.04

ENV TZ=Europe/Stockholm
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# RUN sed -i 's/http:\/\/archive.ubuntu.com\/ubuntu\//http:\/\/ftp.lysator.liu.se\/ubuntu\//' /etc/apt/sources.list

RUN rm /etc/apt/sources.list.d/cuda.list && apt clean && apt update && apt install -q -y --no-install-recommends \
        software-properties-common \
        build-essential \
        python3-pip \
        python-is-python3 \
        net-tools \
        git \
        curl \
        && rm -rf /var/lib/apt/lists/*

RUN pip3 install \
        torch==2.1.2 \
        transformers==4.36
