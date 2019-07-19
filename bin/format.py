# -*- coding: utf-8 -*-
# !/usr/bin/env python
"""
Created on Sat Apr  1 14:35:13 2017

@author: admin
import urllib2,urllib
"""

def format1():
    ts = ['code',
       'abbrev_symbol', 'board_type', 'de_listed_date', 'exchange',
       'industry_code', 'industry_name', 'listed_date', 'market_tplus',
       'order_book_id', 'round_lot', 'sector_code', 'sector_code_name',
       'special_type', 'status', 'symbol', 'trading_hours', 'type']
    for n in ts:
         print('[%s] varchar(20) Null,' % n)

    print('","'.join(ts))

def format():
    sss = """
    [cointype] nvarchar(20) COLLATE SQL_Latin1_General_CP1_CI_AS  NULL,
  [currentprice] real  NULL,
  [futureprice] real  NULL,
  [difference] float(53)  NULL,
  [recordtime] datetime  NULL
    """

    ll =sss.split(",")
    keys = []
    for n in ll:
        key = n[(n.find("[")+1): n.find("]")]
        keys.append(key)
        
    print("','".join(keys))
    
def main():    
    format()
    

if __name__ == '__main__':
    main()
