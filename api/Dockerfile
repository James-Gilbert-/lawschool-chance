FROM python:3.7

#creating directories
RUN mkdir -p /home/project/app/
RUN mkdir -p /home/project/app/models/
WORKDIR /home/project/app

#installing what is needed for flask and sklearn
COPY requirements.txt /home/project/app/
#RUN apt-get -y update && apt-get install -y glib-dev

RUN pip install --no-cache-dir -r requirements.txt
#copy everything over to the working dirs
COPY . /home/project/app/
COPY ./models /home/project/app/models/
