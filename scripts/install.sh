#!/bin/bash
sudo yum update
sudo yum remove python2.7 -y
sudo yum install python37 -y
sudo yum install python-pip -y
sudo yum install apache2 -y
sudo yum install libapache2-mod-wsgi -y
cd /app && pip install --user requirements.txt
