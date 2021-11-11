#!/bin/bash
kill $(pgrep -f 'nohup') >/dev/null 2>&1 &
