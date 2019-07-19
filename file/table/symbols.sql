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
-- Table structure for baseInfo
-- ----------------------------
DROP TABLE [dbo].[baseInfo]
GO
CREATE TABLE [dbo].[baseInfo] (
[ID] int NOT NULL IDENTITY(1,1) NOT FOR REPLICATION ,
[code] varchar(20) NULL ,
[Name] nvarchar(20) NULL ,
[area] varchar(10) NULL ,
[chief] varchar(20) NULL , 
[listing_date] datetime NULL,
[industry] varchar(20) NULL  
)


GO
DBCC CHECKIDENT(N'[dbo].[baseInfo]', RESEED, 1)
GO

-- ----------------------------
-- Indexes structure for table baseInfo
-- ----------------------------

-- ----------------------------
-- Primary Key structure for table baseInfo
-- ----------------------------
ALTER TABLE [dbo].[baseInfo] ADD PRIMARY KEY ([ID]) WITH (FILLFACTOR=90)
GO
