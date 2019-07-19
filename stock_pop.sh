#!/bin/sh
#/usr/bin/ps aux|grep stock| awk '{print $2}'|xargs kill -9
. /etc/profile
. ~/.bash_profile
/usr/bin/nohup /root/anaconda3/bin/python  /root/home/project/stock/bin/stock.py -w 2 & >/dev/null 2>&1


