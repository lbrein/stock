/*
 Navicat Premium Data Transfer

 Source Server         : 116.62.112.161
 Source Server Type    : SQL Server
 Source Server Version : 14003035
 Source Host           : 116.62.112.161:1433
 Source Catalog        : stock
 Source Schema         : dbo

 Target Server Type    : SQL Server
 Target Server Version : 14003035
 File Encoding         : 65001

 Date: 27/08/2018 14:09:36
*/


-- ----------------------------
-- Table structure for train_future_2
-- ----------------------------
IF EXISTS (SELECT * FROM sys.all_objects WHERE object_id = OBJECT_ID(N'[dbo].[train_future_2]') AND type IN ('U'))
	DROP TABLE [dbo].[train_future_2]
GO

CREATE TABLE [dbo].[train_future_2] (
  [ID] int IDENTITY(1,1) NOT FOR REPLICATION NOT NULL,
  [code] varchar(20) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
  [createdate] datetime NULL,
  [price] real NULL,
  [vol] real NULL,
  [mode] int NULL,
  [isopen] int NULL,
  [fee] real NULL,
  [uid] varchar(200) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
  [income] real NULL,
  [rel_price] real NULL,
  [rel_std] real NULL,
  [batchid] varchar(200) COLLATE SQL_Latin1_General_CP1_CI_AS NULL,
  [bullwidth] real NULL,
  [delta] real NULL
)
GO

ALTER TABLE [dbo].[train_future_2] SET (LOCK_ESCALATION = TABLE)
GO


-- ----------------------------
-- Primary Key structure for table train_future_2
-- ----------------------------
ALTER TABLE [dbo].[train_future_2] ADD CONSTRAINT [PK__train_fu__3214EC27328A523F_copy2_copy1] PRIMARY KEY CLUSTERED ([ID])
WITH (PAD_INDEX = OFF, FILLFACTOR = 90, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON)  
ON [PRIMARY]
GO

