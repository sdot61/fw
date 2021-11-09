#!/bin/bash
sudo yum update
sudo amazon-linux-extras enable python3.8
sudo yum install python3.8 -y
sudo yum install python3-pip -y
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.8
cd /app && pip install --user -r requirements.txt
