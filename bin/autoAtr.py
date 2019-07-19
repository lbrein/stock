# -*- coding: utf-8 -*-
#!/usr/bin/env python
"""
Created on Sat Apr  1 14:35:13 2017

@author: admin
     
     窝轮在线检测入口
 
"""

from com.data.msg_weixin import WinxinMsg
from com.base.public import public
from com.data.date_future_Rice import data_future_Rice

def main():
    fw = public.getCmdParam("w:", ("-w", 1))
    if fw == 1:
        obj = WinxinMsg()
        obj.create()

    elif fw==2:
        obj = data_future_Rice()
        for type in [0, 1]:
            obj.autoCreateAtr(type=type)

if __name__=='__main__':
    main()
  
 
    