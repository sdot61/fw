#!/bin/bash
sudo apt-get update
sudo apt-get install python-pip
sudo apt-get install python-flask
sudo apt-get install apache2
sudo apt-get install libapache2-mod-wsgi
cd .. && pip install -r requirements.txt
