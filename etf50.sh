#!/bin/sh
#/usr/bin/ps aux|grep etf50| awk '{print $2}'|xargs kill -9
. ~/.bash_profile;
/usr/bin/nohup /root/anaconda3/bin/python  /root/home/project/stock/bin/etf50.py -w 1 & >/dev/null 2>&1
/usr/bin/nohup /root/anaconda3/bin/python  /root/home/project/stock/bin/etf50.py -w 2 & >/dev/null 2>&1


