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

'name', 'industry', 'area', 'pe', 'outstanding', 'totals',
       'totalAssets', 'liquidAssets', 'fixedAssets', 'reserved',
       'reservedPerShare', 'esp', 'bvps', 'pb', 'timeToMarket', 'undp',
       'perundp', 'rev', 'profit', 'gpr', 'npr', 'holders'],
      dtype='object'
-- ----------------------------
-- Table structure for stockDetail
-- ----------------------------
DROP TABLE [dbo].[stockDetail]
GO
CREATE TABLE [dbo].[stockDetail] (
[ID] int NOT NULL IDENTITY(1,1) NOT FOR REPLICATION ,
[code] varchar(20) Null,
[name] nvarchar(20) Null,
[industry] nvarchar(20) Null,
[area] nvarchar(20) Null,
[pe] real Null,
[outstanding] real Null,
[totals] real Null,
[totalAssets] real Null,
[liquidAssets] real Null,
[fixedAssets] real Null,
[reserved] real Null,
[reservedPerShare] real Null,
[esp] real Null,
[bvps] real Null,
[pb] real Null,
[timeToMarket] date Null,
[undp] real Null,
[perundp] real Null,
[rev] real Null,
[profit] real Null,
[gpr] real Null,
[npr] real Null,
[holders] real Null
)

GO
DBCC CHECKIDENT(N'[dbo].[stockDetail]', RESEED, 1)
GO

-- ----------------------------
-- Indexes structure for table stockDetail
-- ----------------------------

-- ----------------------------
-- Primary Key structure for table stockDetail
-- ----------------------------
ALTER TABLE [dbo].[stockDetail] ADD PRIMARY KEY ([ID]) WITH (FILLFACTOR=90)
GO
