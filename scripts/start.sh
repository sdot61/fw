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
sudo mv -f ./flask-app.service /etc/systemd/system
sudo systemctl daemon-reload
sudo systemctl enable flask-app.service
sudo systemctl start flask-app.service
