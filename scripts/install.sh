#!/bin/bash
sudo yum update
sudo yum install python-pip -y
sudo yum install apache2 -y
sudo yum install libapache2-mod-wsgi -y
echo pwd
pip install -r requirements.txt
