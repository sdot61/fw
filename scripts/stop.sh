#!/bin/bash
kill $(pgrep -f 'python3.8 ./application.py') >/dev/null 2>&1 &
