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
DROP TABLE [dbo].[maticIndex]
GO
CREATE TABLE [dbo].[maticIndex] (
[ID] int NOT NULL IDENTITY(1,1) NOT FOR REPLICATION ,
[date] int Null ,
[index]  int NULL ,
[modifytime] datetime NUll 
)

GO
DBCC CHECKIDENT(N'[dbo].[maticIndex]', RESEED, 0)
GO

-- ----------------------------
-- Indexes structure for table maticIndex
-- ----------------------------

-- ----------------------------
-- Primary Key structure for table maticIndex
-- ----------------------------
ALTER TABLE [dbo].[maticIndex] ADD PRIMARY KEY ([ID]) WITH (FILLFACTOR=90)
GO
