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
-- Table structure for train_future
-- ----------------------------
DROP TABLE [dbo].[train_future]
GO
CREATE TABLE [dbo].[train_future] (
[ID] int NOT NULL IDENTITY(1,1) NOT FOR REPLICATION ,
[code] varchar(20) Null,
[createdate] datetime Null,
[price] real Null,
[vol] real Null,
[mode] int Null,
[isopen] int Null,
[fee] real Null,
[uid] varchar(200) Null,
[income] real Null,
[rel_price] real null,
[rel_std] real null
)

GO
DBCC CHECKIDENT(N'[dbo].[train_future]', RESEED, 1)
GO

-- ----------------------------
-- Indexes structure for table train_future
-- ----------------------------

-- ----------------------------
-- Primary Key structure for table train_future
-- ----------------------------
ALTER TABLE [dbo].[train_future] ADD PRIMARY KEY ([ID]) WITH (FILLFACTOR=90)
GO
