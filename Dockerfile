FROM jupyter/base-notebook:python-3.10.4

WORKDIR /home/climatology/

RUN pip install --upgrade pip

COPY requirements.txt .

RUN pip install -r requirements.txt