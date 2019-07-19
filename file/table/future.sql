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
-- Table structure for future_baseInfo
-- ----------------------------
DROP TABLE [dbo].[future_baseInfo]
GO
CREATE TABLE [dbo].[future_baseInfo] (
[ID] int NOT NULL IDENTITY(1,1) NOT FOR REPLICATION ,
[code] varchar(20) Null,
[ratio] real Null,
[symbol] nvarchar(20) Null,
[exchange] varchar(20) Null,
[margin_rate] real Null,
[contract_multiplier] real null,
[product] varchar(20) Null 
)

GO
DBCC CHECKIDENT(N'[dbo].[future_baseInfo]', RESEED, 1)
GO

-- ----------------------------
-- Indexes structure for table future_baseInfo
-- ----------------------------

-- ----------------------------
-- Primary Key structure for table future_baseInfo
-- ----------------------------
ALTER TABLE [dbo].[future_baseInfo] ADD PRIMARY KEY ([ID]) WITH (FILLFACTOR=90)
GO
