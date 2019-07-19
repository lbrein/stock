# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 11:32:22 2017
@author: lbrein
sql数据库映射类
"""

from com.base.public import baseSql, public
import pandas as pd


class future_code(baseSql):
    tablename = "future_code"
    keylist = up_struct = ['code', 'name', 'de_listed_date', 'exchange', 'market_tplus', 'margin_rate', 'symbol',
                           'order_book_id', 'underlying_symbol', 'contract_multiplier', 'listed_date', 'trading_hours',
                           'type', 'maturity_date', 'main_date', 'used_date', 'de_used_date', 'used_volume']

    Chinese_List = ['symbol']

    def getMap(self, start=None, end = None):
        if end is None: end = public.getDate()
        sql = """
            declare @s datetime, @e datetime
			set @s = '%s'
			set @e = '%s'
			select code, name, 
			case when used_date < @s then @s else used_date end as startdate, 
			case when de_used_date > @e then @e else de_used_date end as enddate
			from %s 
			where  used_date is not null and ((@s BETWEEN  used_date and de_used_date ) or (@e BETWEEN  used_date and de_used_date) or (@s <= used_date and  @e >= de_used_date))
			ORDER BY name, code
        """ % (start, end, self.tablename)
        #print(sql)
        self.keylist = ['code', 'name', 'startdate', 'enddate']
        return [d for d in self.execSql(sql)]

class future_train_source(baseSql):
    tablename = "train_future_2"
    keylist = ['mode', 'diff', 'bias', 'jump_n', 'isup', 'isbig', 'iscross', 'isatr', 'income']

    def getdata(self):
        sql = """
            select 
              mode,
              diff,
              sign(mode) * bias as bias,
              sign(mode) * jump_n as jump_n,  
              sign(mode) * isup as isup,
              - sign(mode) * isbig as isbig,
              sign(mode) * iscross as iscross,
              - sign(mode) * isatr as isatr,
              income 
			  from
			      (select 
              max(case when isopen<> 0 then mode else -10 end) as mode,  
              datediff(day,min(createdate),max(createdate)) as diff,
              max(case when isopen<> 0 then shift else -100 end) as bias,  
                max(case when isopen<> 0 then -1 * delta else -100 end) as jumpback,  
                max(case when isopen<> 0 then widthdelta else -100 end) as jump_n,  
                max(case when isopen<> 0 then p_h else -100 end) as isup,  
                max(case when isopen<> 0 then -1 * p_l else -100 end) as isbig,
                max(case when isopen<> 0 then mastd else -100 end) as iscross,
                max(case when isopen<> 0 then -1 * macd2d else -100 end) as isatr,
                
              case when sum(fee) = 0 then 0 else round(sum(income)/sum(fee),1) end as income 
              from %s 
              
			  group by batchid
              having count(*) >1) a
        
        """ % self.tablename
        return pd.DataFrame([d for d in self.execSql(sql)], columns=self.keylist)


#  模型线性回归训练表
class future_model(baseSql):
    tablename = "model"
    keylist = up_struct = ['name', 'learnmethod', 'method', 'score', 'score_sd', 'count', 'c0', 'c1', 'c2', 'c3',
                           'c0_sd', 'c1_sd', 'c2_sd', 'c3_sd', 'c4', 'c4_sd', 'c5', 'c5_sd', 'c6', 'c6_sd', 'c7',
                           'c7_sd', 'c8',
                           'c8_sd', 'c9', 'c9_sd', 'cond', 'cluster', 'nameindex', 'memo']
