#!/bin/bash
alias python=python3.8
alias pip=pip3
sudo yum update -y
sudo amazon-linux-extras install epel -y
sudo yum install npm -y
sudo npm install pm2@latest -g
cd /app
pip3 install --user -r requirements.txt
python3.8 -m pip install flask
#export PORT=80
sudo nohup python3.8 ./application.py >log.txt 2>&1 &
#pm2 start ./application.py >~log.txt 2>&1 &
