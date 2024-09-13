FROM python:3.8.19-slim
RUN apt update -y && apt install -y awscli build-essential
WORKDIR /app
COPY . /app

RUN pip install --upgrade pip
RUN pip install python-dotenv
RUN pip install -r requirements.txt

CMD [ "python3","main.py" ]

