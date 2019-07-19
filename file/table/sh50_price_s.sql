/*
 Navicat SQL Server Data Transfer

 Source Server         : 116.62.112.161
 Source Server Type    : SQL Server
 Source Server Version : 14003045
 Source Host           : 116.62.112.161:1433
 Source Catalog        : stock
 Source Schema         : dbo

 Target Server Type    : SQL Server
 Target Server Version : 14003045
 File Encoding         : 65001

 Date: 15/11/2018 18:10:59
*/


-- ----------------------------
-- Table structure for sh50_price_s
-- ----------------------------
IF EXISTS (SELECT * FROM sys.all_objects WHERE object_id = OBJECT_ID(N'[dbo].[sh50_price_s]') AND type IN ('U'))
	DROP TABLE [dbo].[sh50_price_s]
GO

CREATE TABLE [dbo].[sh50_price_s] (
  [ID] int  IDENTITY(1,1) NOT FOR REPLICATION NOT NULL,
  [code] varchar(20) COLLATE SQL_Latin1_General_CP1_CI_AS  NULL,
  [bid_vol] real  NULL,
  [bid] real  NULL,
  [price] real  NULL,
  [ask] real  NULL,
  [ask_vol] real  NULL,
  [power] real  NULL,
  [sopen] real  NULL,
  [currentTime] datetime  NULL,
  [sign] int  NULL,
  [state] varchar(20) COLLATE SQL_Latin1_General_CP1_CI_AS  NULL,
  [high] real  NULL,
  [low] real  NULL,
  [volume] real  NULL,
  [amount] real  NULL,
  [createTime] datetime  NULL
)
GO

ALTER TABLE [dbo].[sh50_price_s] SET (LOCK_ESCALATION = TABLE)
GO


-- ----------------------------
-- Primary Key structure for table sh50_price_s
-- ----------------------------
ALTER TABLE [dbo].[sh50_price_s] ADD CONSTRAINT [PK__sh50_pri__3214EC276911DB94_copy2] PRIMARY KEY CLUSTERED ([ID])
WITH (PAD_INDEX = OFF, FILLFACTOR = 90, STATISTICS_NORECOMPUTE = OFF, IGNORE_DUP_KEY = OFF, ALLOW_ROW_LOCKS = ON, ALLOW_PAGE_LOCKS = ON)  
ON [PRIMARY]
GO

