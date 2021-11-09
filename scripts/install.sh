#!/bin/bash
sudo yum update
sudo amazon-linux-extras enable python3.8
sudo yum install python3.8 -y
alias python=python3.8
cd /app && pip install --user -r requirements.txt
