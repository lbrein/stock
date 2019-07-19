/*
Navicat SQL Server Data Transfer

Source Server         : bc_online
Source Server Version : 140000
Source Host           : 116.62.132.182:1433
Source Database       : scrapy
Source Schema         : dbo

Target Server Type    : SQL Server
Target Server Version : 140000
File Encoding         : 65001

Date: 2018-03-06 18:28:50
*/


-- ----------------------------
-- Table structure for maticIndex
-- ----------------------------
DROP TABLE [dbo].[sh50_daily]
GO
CREATE TABLE [dbo].[sh50_daily] (
[ID] int NOT NULL IDENTITY(1,1) NOT FOR REPLICATION ,
[code] varchar(25) Null, 
[date] date Null,
[pre_close] real Null,
[open] real Null,
[high] real Null,
[low] real Null,
[close] real Null,
[volume] real Null,
[amt] real Null,
[dealnum] real Null,
[chg] real Null,
[pct_chg] real Null,
[swing] real Null,
[vwap] real Null,
[oi] real Null,
[oi_chg] real Null,
[adjfactor] real Null
)

GO
DBCC CHECKIDENT(N'[dbo].[sh50_daily]', RESEED, 1)
GO

-- ----------------------------
-- Indexes structure for table sh50_daily
-- ----------------------------

-- ----------------------------
-- Primary Key structure for table sh50_daily
-- ----------------------------
ALTER TABLE [dbo].[sh50_daily] ADD PRIMARY KEY ([ID]) WITH (FILLFACTOR=90)
GO

