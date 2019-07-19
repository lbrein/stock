#!/bin/sh
/usr/bin/ps aux|grep python| awk '{print $2}'|xargs kill -9
/usr/bin/nohup /root/anaconda3/bin/python  /root/home/project/stock/bin/sh50_inform.py & >/dev/null 2>&1
/usr/bin/nohup /root/anaconda3/bin/python  /root/home/project/stock/bin/etf_order.py & >/dev/null 2>&1
/usr/bin/nohup /root/anaconda3/bin/python  /root/home/project/stock/com/model/model_sh50_MA.py & >/dev/null 2>&1

exit 0





