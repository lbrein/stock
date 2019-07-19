# -*- coding: utf-8 -*-
"""
Created on  3-22  2017
@author: lbrein
整理过的基础类库
 从配置文件获取连接定义，位于 config.ini
"""
import sys
import re
import pymysql, pymssql
import logging
from logging.handlers import TimedRotatingFileHandler
import configparser
import os
import datetime
import time
from pymongo import MongoClient
import getopt
import pandas as pd

filepath = re.sub("\\\\", "/", str(os.path.dirname(__file__)))
public_basePath = filepath[0:(filepath.rfind("/com/") + 1)]
Public_parser = configparser.ConfigParser()
Public_parser.read(public_basePath + "config.ini")

class config_ini:
    @staticmethod
    def get(key, session="db", default=""):
        try:
            return Public_parser.get(session, key)
        except:
            return default

    @staticmethod
    def Full(session):
        for opt in Public_parser.items(session):
            yield opt

# 连接mongondb
class con:
    @staticmethod
    def connect(host="localhost", dbname="hr"):
        if host == "" or host == "localhost":
            host = config_ini.get("mongodb.host")
        if dbname == "" or dbname == "hr":
            dbname = config_ini.get("mongodb.db")
        try:
            client = MongoClient(host, 27017, connect=False)
            db = client[dbname]
            if db: print("%s 数据库 %s ,%s 连接成功!" % ("mongodb", host, dbname))
            return client, db
        except:
            return None, None

# 链接到mysql
class conm:
    @staticmethod
    def connect():
        sqltype = config_ini.get("sqltype")

        if sqltype is None: sqltype = "mysql"

        host = config_ini.get(sqltype + ".host")
        dbname = config_ini.get(sqltype + ".db")
        username = config_ini.get(sqltype + ".username")
        psd = config_ini.get(sqltype + ".password")

        try:
            if sqltype == "mysql":
                conn = pymysql.connect(host, username, psd, dbname, charset='utf8')
            elif sqltype == "mssql":
                conn = pymssql.connect(host=host, user=username, password=psd, database=dbname, charset='utf8')

            cursor = conn.cursor()
            if cursor: print("%s 数据库 %s ,%s 连接成功!" % (sqltype, host, dbname))
            return conn, cursor

        except:
            print("mssql %s 数据库连接失败" % host)
            return None, None

    @staticmethod
    def sql_connect(host, db, user, psd):
        try:
            conn = pymssql.connect(host=host, user=user, password=psd, database=db, charset='utf8')
            cursor = conn.cursor()
            if cursor: print("sqlserver数据库 %s ,%s 连接成功!" % (host, db))
            return conn, cursor
        except:
            print("数据库连接失败")
            return None, None

class public:
    public_mongodb_client, public_db = None, None

    @staticmethod
    def getDate(diff=0, start=None):
        if start is None:
            return (datetime.datetime.now() + datetime.timedelta(days=diff)).strftime("%Y-%m-%d")
        else:
            dd = datetime.datetime.strptime(start, '%Y-%m-%d')
            return (dd + datetime.timedelta(days=diff)).strftime("%Y-%m-%d")

    @staticmethod
    def getWeekDay(date=None):
        if date is not None:
            f = '%Y-%m-%d %H:%M:%S.%f' if date.find(".") > -1 else '%Y-%m-%d %H:%M:%S'
            dd = datetime.datetime.strptime(date, f)
        else:
            dd = datetime.datetime.now()
        return dd.weekday()

    @staticmethod
    def getDatetime(diff=0, hours=0, minutes=0, style='%Y-%m-%d %H:%M:%S', start=None):
        if start is None:
            return (datetime.datetime.now() + datetime.timedelta(days=diff, hours=hours, minutes=minutes)).strftime(style)
        else:
            tt = datetime.datetime.strptime(start, style)
            return (tt + datetime.timedelta(days=diff, hours=hours, minutes=minutes)).strftime(style)

    @staticmethod
    def getStamp(diff=0):
        return time.time() + diff * 24 * 3600

    @staticmethod
    def parseStamp(date, diff=0):
        f = '%Y-%m-%d %H:%M:%S.%f' if date.find(".") > -1 else '%Y-%m-%d %H:%M:%S'
        timeArray = time.strptime(date, f)
        return int(time.mktime(timeArray)) + diff * 3600

    @staticmethod
    def getTime(diff=0, style="%H%M%S"):
        return (datetime.datetime.now() + datetime.timedelta(days=diff)).strftime(style)

    @staticmethod
    def timeDiff(t0, t1):
        a, b = 0, 0
        if type(t0) == str:
             f = '%Y-%m-%d %H:%M:%S.%f' if t0.find(".") > -1 else '%Y-%m-%d %H:%M:%S'
             if len(t0)< 11:
                    t0, t1 = public.getDate()+' '+t0, public.getDate()+' '+t1

             a = time.mktime(time.strptime(t0, f))
             b = time.mktime(time.strptime(t1, f))

        elif type(t0) == datetime.datetime:
            a = time.mktime(t0.timetuple())
            b = time.mktime(t1.timetuple())
        return (a - b)

    @staticmethod
    def dayDiff(t0, t1):
        # print(type(t0))
        a, b = 0, 0
        if type(t0) == str:
            f = '%Y-%m-%d'
            a = time.mktime(time.strptime(t0, f))
            b = time.mktime(time.strptime(t1, f))
        elif type(t0) == datetime.datetime:
            a = time.mktime(t0.timetuple())
            b = time.mktime(t1.timetuple())

        return (a - b) / 24 / 3600

    @staticmethod
    def str_time(str):
        format = '%Y-%m-%d %H:%M:%S'
        if str.find(".") > 0: format = '%Y-%m-%d %H:%M:%S.%f'
        return time.mktime(time.strptime(str, format))

    @staticmethod
    def str_date(str, format = None):
        if format == None:
            format = '%Y-%m-%d %H:%M:%S'
            if str.find(".") > 0: format = '%Y-%m-%d %H:%M:%S.%f'
        return datetime.datetime.strptime(str, format)

    @staticmethod
    def parsestamptime(ss):
        now_localtime = time.localtime(ss)
        return time.strftime('%Y-%m-%d %H:%M:%S', now_localtime)

    @staticmethod
    def getMonthDay(date=None):
        if date is None:
            date = public.getDate()

        y, m, d = (int(c) for c in date.split('-'))
        d0, d1 = datetime.datetime(y, m, 1), datetime.datetime(y, m, d)
        begin, end = int(d0.strftime("%W")), int(d1.strftime("%W"))
        w0, w = d0.weekday(), d1.weekday()
        return end - begin + 1 + (-1 if w0 in [5, 6] else 0), w + 1


    @staticmethod
    def monthDiff(d1, d2):
        if d1 > d2: return 0
        y1, m1 = int(d1.split('-')[0]), int(d1.split('-')[1])
        y2, m2 = int(d2.split('-')[0]), int(d2.split('-')[1])
        d = (y2 - y1) * 24 + (m2 - m1 if m2 >= m1 else 12 + m2 - m1) + 1
        return d

    @staticmethod
    def parseDate(ss):
        pass

    @staticmethod
    def parseTime(ss, format=None, style=None):
        if format is None:
            format = '%Y-%m-%d %H:%M:%S'
            if str(ss).find(".") > 0: format = '%Y-%m-%d %H:%M:%S.%f'

        if style is None:
            style = format

        tmp = time.strptime(ss, format)
        return time.strftime(style, tmp)

    @staticmethod
    def stamptoTime(ss):
        # time_local = time.localtime(ss)
        # 转换成新的时间格式(2016-05-05 20:28:54)
        dt = time.strftime("%Y-%m-%d %H:%M:%S", (ss))
        return dt

    @staticmethod
    def getBasePath():
        return public_basePath

    # 迭代器，1000
    @staticmethod
    def eachPage(docs, pc=1000):
        c = len(docs)
        pages = c // pc + 1
        for i in range(pages):
            s, e = i * pc, (i + 1) * pc
            if e > c: e = c
            yield docs[s:e]

    @staticmethod
    def getCmdParam(define, key):
        # key为tuple,则带默认值
        opts, args = getopt.getopt(sys.argv[1:], define)
        if isinstance(key, tuple):
            for op, value in opts:
                if op == key[0]:
                    if isinstance(key[1], int):  # 判断是否为数字
                        return int(value)
                    else:
                        return value
            return key[1]
        else:
            for op, value in opts:
                if op == key: return value
            return ""

    @staticmethod
    def to_csv(docs, folder, filename, headers=None):
        if not docs:
            return None

        if not headers:
            headers = docs[0].keys()

        cvs = csvFile(folder, filename=filename, headers=headers)
        for doc in docs:
            cvs.writeLine(doc)
        cvs.close()


class Logger:

    def __init__(self):
        self._set()

    def _set(self, path="stock_info.log", clevel=logging.DEBUG, Flevel=logging.DEBUG):
        self.filepath = self.getPath()
        self.logger = logging.getLogger(path)
        self.logger.setLevel(logging.DEBUG)
        fmt = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')
        # 设置CMD日志
        sh = None
        if not self.logger.handlers:
            sh = logging.StreamHandler()
            sh.setFormatter(fmt)
            sh.setLevel(clevel)
            # 设置文件日志
            # fh = logging.FileHandler(self.filepath)
            fh = TimedRotatingFileHandler(self.filepath, 'D', 1, 0)
            # fh = logging.handlers
            fh.setFormatter(fmt)
            fh.setLevel(Flevel)
            self.logger.addHandler(sh)
            self.logger.addHandler(fh)
        else:
            self.logger.removeHandler(sh)

    def getPath(self):
        # 返回每天的日志
        date = public.getDate()
        filepath = public_basePath + "/log/stock_log_" + date + ".log"
        return filepath

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warn(self, message):
        self.logger.warn(message)

    def error(self, message):
        self.logger.error(message)

    def cri(self, message):
        self.logger.critical(message)

    itemCount = 0
    def debugT(self, str, n=1000):
        self.itemCount += 1
        if self.itemCount % n == 0:
            print(self.itemCount, str)

"""
   全局变量
   数据库连接
   配置文件读取
"""

logger = Logger()
public_mongodb_client, public_db = None, None


# mongodb通用类
class baseObject(object):
    colName = "default"
    source = ""
    keylist = []

    def __init__(self, client=None, db=None, isPublic=True):
        global public_mongodb_client, public_db
        if isPublic:
            if public_mongodb_client is None:
                public_mongodb_client, public_db = con.connect()

            self.mongodb_client, self.public_db = public_mongodb_client, public_db

        elif client is None or db is None:
            self.mongodb_client, self.public_db = con.connect()

        else:
            self.mongodb_client, self.public_db = client, db

        self.col = self.getCollection(self.colName)

    def getCollection(self, colName):
        col = self.public_db.get_collection(colName)
        if not col:
            col = self.public_db.create_collection(colName)
        return col

    def close(self):
        self.public_db.close()

    def set(self, item):
        k = 0
        doc = {}
        for key in self.keylist:
            try:
                doc[key] = item[k]
                k = k + 1
            except:
                logger.error(sys.exc_info())
                continue
        return doc

    def find(self, sql):
        return self.col.find(sql)

    def save(self, doc):
        tmp = self.col.insert_one(doc)
        doc["_id"] = tmp.inserted_id
        return doc

    def empty(self, filter={}):
        self.col.remove(filter)

    def saveall(self, docs):
        self.col.insert_many(docs)

    def remove(self, doc):
        self.col.delete_one(doc)

    def log(self, action, info):
        str = self.colName
        if action == 0 or action == "import":
            str += "导入文件成功,%s" % info
        elif action == 1 or action == "update":
            str += "修改文件成功,%s" % info
        elif action == 2 or action == "delete":
            str += "删除文件成功,%s" % info
        else:
            str += "%s成功, %s" % (action, info)
        logger.info(str)


public_conm, public_cursor = None, None

class baseSql():
    tablename = ""
    keylist = []  # 下行字段列表
    up_struct = []  # 上行字段篱笆
    Chinese_List = []  # 中文字段篱笆
    sql_update = ""
    perserve = ["open", "date", "close", "count", "start", "end", "order"]
    self_conm, self_cur = None, None
    paramMap = {}
    configMap = {}

    def __init__(self, con=None, cursor=None):
        global public_conm, public_cursor
        if con is None or cursor is None:
            if public_conm is None:
                self.self_conm, self.self_cur = conm.connect()
                public_conm, public_cursor = self.self_conm, self.self_cur
                print('sub connect', self.tablename)
            else:
                print('public connect', self.tablename)
                self.self_conm, self.self_cur = public_conm, public_cursor
        else:
            self.self_conm, self.self_cur = con, cursor


    def set(self, item):
        k = 0
        doc = {}
        for key in self.keylist:
            try:
                doc[key] = item[k]
                k = k + 1
            except:
                logger.error(sys.exc_info())
                continue
        return doc

    def closecur(self):
        global public_conm, public_cursor

        self.self_conm.close()
        public_conm = self.self_conm = None
        public_cursor = None
        print('close connect', self.tablename)

    def execSql(self, sql, isJson=True):
        self.self_cur.execute("BEGIN tran " + sql+ " COMMIT tran")
        if isJson:
            for doc in [self.set(rec) for rec in self.self_cur.fetchall()]:
                yield doc
        else:
            for rec in self.self_cur.fetchall():
                yield rec

    def execOne(self, sql, isJson=True):
        self.self_cur.execute(sql)
        doc = self.self_cur.fetchone()

        if doc is not None and isJson:
            return self.set(doc)
        else:
            return doc

    # 返回df
    def df(self, sql):
        res = self.execSql(sql)
        return pd.DataFrame([doc for doc in res], columns=self.keylist)

    def empty(self, filter='1=1'):
        sql = "delete from %s where id>0 and %s" % (self.tablename, filter)
        self.self_cur.execute(sql)
        self.self_conm.commit()
        # self.closecur()
        print("%s 已清空 " % self.tablename)

    def update(self, sql=sql_update):
        self.self_cur.execute(sql)
        self.self_conm.commit()
        # self.closecur()

    def getNames(self):
        keys = []
        for item in self.up_struct:
            if item in self.perserve:
                item = "[%s]" % item
            keys.append(item)
        return ",".join(keys)

    def getValues(self, doc):
        str_value = ""
        for key in self.up_struct:
            # 中文输入添加N
            if self.Chinese_List and key in self.Chinese_List and key in doc.keys():
                str_value += ",N'%s'" % str(doc[key]).replace("'", "''")
            elif key in doc.keys():
                str_value += ",'%s'" % str(doc[key]).replace("'", "''").replace("inf", "0").replace("nan", "0")
            else:
                str_value += ", Null"
        return str_value[1:]

    # 插入
    def insert(self, doc):
        sql = "BEGIN tran  insert into dbo.%s  (%s) values(%s) COMMIT tran" % (self.tablename, self.getNames(), self.getValues(doc))
        self.self_cur.execute(sql)
        self.self_conm.commit()

    # 插入并返回ID
    def re_insert(self, doc):
        sql = "insert into dbo.%s (%s) values('%s') SELECT @@IDENTITY" % (self.tablename, self.getNames(), self.getValues(doc))
        self.self_cur.execute(sql)
        res = self.self_cur.fetchone()
        self.self_conm.commit()
        return res[0]

        # 批量添加
    def insertAll(self, docs):
        if len(docs) == 0: return None
        sql = "insert into dbo.%s (%s) " % (self.tablename, self.getNames())

        for doc in docs:
            sql = sql + "select %s  union all " % self.getValues(doc)
        sql = sql[:(len(sql) - 10)]
        #print(sql)
        try:
            self.self_cur.execute("BEGIN tran " + sql + "  COMMIT tran")
            self.self_conm.commit()
            return True
        except:
            logger.error(sys.exc_info())
            logger.error(sql)
            return False

    def parseSql(self, sql, stage='stage'):
        sql_r = sql
        # 先处理paramMap的参数
        maps = self.paramMap
        for key in maps.keys():
            sql_r = sql_r.replace(key, str(maps[key]))

        for name, value in config_ini.Full(stage):
            ary = name.split(".")
            key = ary[len(ary) - 1]
            sql_r = sql_r.replace("$%s" % key, value)
        return sql_r

import csv
import numpy as np

class csvFile:
    filename = ""
    csv_file = None
    writer = None
    headers = None

    def __init__(self, folder="", filename="test.csv", headers=None):
        # if folder != "" and (folder.find("/") < 0 or folder.find(":") < 0):
        #    folder = folder
        f = folder
        # if folder.find(":") == -1:
        #    f = public_basePath + "/" + f
        if not os.path.exists(f):
            os.mkdir(f)

        self.folder = f
        self.filename = filename
        self.headers = headers

    def openfile(self, doc):
        # 检查并删除文件
        filepath = self.folder + "/" + self.filename
        csv_file = open(filepath, 'w', newline='')
        # 添加表头
        if not self.headers:
            self.headers = [k for k in doc.keys()]

        writer = csv.DictWriter(csv_file, fieldnames=self.headers)
        writer.writeheader()
        self.writer = writer
        self.csv_file = csv_file
        return writer

    def read(self):
        filepath = self.folder + "/" + self.filename
        self.csv_file = open(filepath, 'r')
        reader = csv.reader(self.csv_file)
        for line in reader:
            yield line

    def writeLine(self, doc):
        writer = self.writer
        if not writer:
            writer = self.openfile(doc)
        writer.writerow(doc)

    def npwrite(self, ary):
        filepath = self.folder + "/" + self.filename
        np.savetxt(filepath, ary, delimiter=',', header="")

    def close(self):
        self.csv_file.close()


import ftplib as fp
import platform

class ftpFile(object):
    def __init__(self):
        ftp_host = config_ini.get("ftp.host")
        ftp_port = int(config_ini.get("ftp.port"))
        ftp_username = config_ini.get("ftp.username")
        ftp_password = config_ini.get("ftp.password")
        self.slash = "\\" if platform.platform().find("window") > 0 else "//"
        self.Ftp = fp.FTP()
        self.Ftp.set_pasv(1)
        self.Ftp.connect(ftp_host, ftp_port)
        self.Ftp.login(ftp_username, ftp_password)

    def list(self):
        print(self.Ftp.retrlines('LIST'))

    def download(self, folder, filename):
        file_remote = filename
        file_local = folder + self.slash + filename
        bufsize = 1024  # 设置缓冲器大小
        file = open(file_local, 'wb')
        self.Ftp.retrbinary('RETR %s' % file_remote, file.write, bufsize)
        file.close()
        return file_local

    def upload(self, folder, filename):
        '''以二进制形式上传文件'''

        file_remote = filename
        file_local = folder + self.slash + filename

        bufsize = 1024  # 设置缓冲器大小
        file = open(file_local, 'rb')
        self.Ftp.storbinary('STOR ' + file_remote, file, bufsize)
        file.close()

    def close(self):
        self.Ftp.close()


def main():
    # 控制开关
    ftp = ftpFile()
    ftp.list()

    # folder = "E:\stock\guangfa"
    # ftp.download(folder, 'asset.20180725.csv')
    # ftp.upload(folder, "instr_sh50.20180720134311573.csv")


if __name__ == '__main__':
    main()

