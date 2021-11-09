#!/bin/bash
pid=$(cat /tmp/app.pid)

if [pid>0]
then
  kill -9 pid
fi;
