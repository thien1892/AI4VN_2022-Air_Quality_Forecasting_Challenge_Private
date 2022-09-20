FROM python:3.9

WORKDIR /bkav_aivn2022

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .