#!/bin/bash
alias python=python3.8
alias pip=pip3
sudo yum update -y
cd /app && pip install --user -r requirements.txt
