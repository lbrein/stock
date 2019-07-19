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
-- Table structure for orderRecord
-- ----------------------------
--DROP TABLE [dbo].[orderRecord]
GO
CREATE TABLE [dbo].[orderRecord] (
[ID] int NOT NULL IDENTITY(1,1) NOT FOR REPLICATION ,
[code] varchar(20) Null,
[market] varchar(20) Null,
[mode] int null,
[type] varchar(20) Null,
[price] real Null,
[volume] real Null,
[amount] real Null,
[account] int  Null,
[company]  nvarchar(20) null, 
[createTime] datetime Null,
[status] real Null,
[returnid] int Null
)

GO
DBCC CHECKIDENT(N'[dbo].[orderRecord]', RESEED, 1)
GO

-- ----------------------------
-- Indexes structure for table orderRecord
-- ----------------------------

-- ----------------------------
-- Primary Key structure for table orderRecord
-- ----------------------------
ALTER TABLE [dbo].[orderRecord] ADD PRIMARY KEY ([ID]) WITH (FILLFACTOR=90)
GO
