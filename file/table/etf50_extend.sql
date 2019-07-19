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
DROP TABLE [dbo].[sh50_extend]
GO
CREATE TABLE [dbo].[sh50_extend] (
[ID] int NOT NULL IDENTITY(1,1) NOT FOR REPLICATION ,
[date] date Null,
[open] real Null,
[high] real Null,
[low] real Null,
[close] real Null,
[volume] real Null,
[fq_ratio] real Null,
[qfq_ratio] real Null,
[nfq] real Null,
[qfq] real Null,
[price_2016] real Null,
[price_2017] real Null,
[range] real Null,
[range_244] real Null
)

GO
DBCC CHECKIDENT(N'[dbo].[sh50_extend]', RESEED, 1)
GO

-- ----------------------------
-- Indexes structure for table sh50_extend
-- ----------------------------

-- ----------------------------
-- Primary Key structure for table sh50_extend
-- ----------------------------
ALTER TABLE [dbo].[sh50_extend] ADD PRIMARY KEY ([ID]) WITH (FILLFACTOR=90)
GO
