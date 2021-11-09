#!/bin/bash
sudo yum update
sudo yum install python-pip
sudo yum install python-flask
sudo yum install apache2
sudo yum install libapache2-mod-wsgi
cd .. && pip install -r requirements.txt
