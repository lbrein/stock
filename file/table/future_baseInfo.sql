/*
 Navicat Premium Data Transfer

 Source Server         : 116.62.112.161
 Source Server Type    : SQL Server
 Source Server Version : 14003192
 Source Host           : 116.62.112.161:1433
 Source Catalog        : stock
 Source Schema         : dbo

 Target Server Type    : SQL Server
 Target Server Version : 14003192
 File Encoding         : 65001

 Date: 18/07/2019 22:55:45
*/


-- ----------------------------
-- Table structure for future_baseInfo
-- ----------------------------
IF EXISTS (SELECT * FROM sys.all_objects WHERE object_id = OBJECT_ID(N'[dbo].[future_baseInfo]') AND type IN ('U'))
	DROP TABLE [dbo].[future_baseInfo]
GO

CREATE TABLE [dbo].[future_baseInfo] (
  [ID] int  IDENTITY(1,1) NOT FOR REPLICATION NOT NULL,
  [code] varchar(20) COLLATE SQL_Latin1_General_CP1_CI_AS  NULL,
  [ratio] real  NULL,
  [symbol] nvarchar(20) COLLATE SQL_Latin1_General_CP1_CI_AS  NULL,
  [exchange] varchar(20) COLLATE SQL_Latin1_General_CP1_CI_AS  NULL,
  [margin_rate] real  NULL,
  [contract_multiplier] real  NULL,
  [product] varchar(20) COLLATE SQL_Latin1_General_CP1_CI_AS  NULL,
  [nightEnd] time(7)  NULL,
  [tick_size] real  NULL,
  [lastPrice] real  NULL,
  [lastVolume] real  NULL,
  [ctp_symbol] varchar(20) COLLATE SQL_Latin1_General_CP1_CI_AS  NULL,
  [ctp_main] varchar(20) COLLATE SQL_Latin1_General_CP1_CI_AS  NULL,
  [bulltype] int  NULL,
  [pstatus] int  NULL,
  [isUsed] int  NULL,
  [klinetype] varchar(10) COLLATE SQL_Latin1_General_CP1_CI_AS  NULL,
  [trend] int  NULL,
  [quickkline] varchar(10) COLLATE SQL_Latin1_General_CP1_CI_AS  NULL,
  [ratio_today] real  NULL
)
GO

ALTER TABLE [dbo].[future_baseInfo] SET (LOCK_ESCALATION = TABLE)
GO


-- ----------------------------
-- Primary Key structure for table future_baseInfo
-- ----------------------------
ALTER TABLE [dbo].[future_baseInfo] ADD CONSTRAINT [PK__future_b__3214EC275E4A963B] PRIMARY KEY CLUSTERED ([ID])
WITH (PAD_INDEX = OFF, FILLFACTOR = 90, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON)  
ON [PRIMARY]
GO

