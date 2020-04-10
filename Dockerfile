FROM ubuntu:18.04

RUN apt-get update && apt-get install -y python3 python3-pip wget
WORKDIR /repo
COPY requirements-cpu.txt requirements-cpu.txt
RUN pip3 install -r requirements-cpu.txt -f https://download.pytorch.org/whl/torch_stable.html
