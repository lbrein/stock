# -*- coding: utf-8 -*-
# !/usr/bin/env python
"""
Created on Sat Apr  1 14:35:13 2017

@author: admin

      sh50自动记录

"""

from com.model.model_stock_pop import model_stock_pop

from com.base.public import public

if __name__ == '__main__':
    fw = public.getCmdParam("w:", ("-w", 1))
    if fw == 1:
        obj = model_stock_pop()
        obj.pool_filter()

    elif fw == 2:
        obj = model_stock_pop()
        obj.control()

    elif fw == 3:
        obj = model_stock_pop()
        obj.crontab()

