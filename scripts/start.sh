#!/bin/bash
alias python=python3.8
alias pip=pip3
sudo yum update -y
cd /app
pip3 install --user -r requirements.txt
pip3 install forever

python3.8 -m pip install flask
#export PORT=80
#sudo  python3.8 -m forever.run ./application.py >/dev/null 2>&1 &
sudo python3.8 ./application.py >/dev/null 2>&1 &
