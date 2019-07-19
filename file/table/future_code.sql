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

 Date: 18/07/2019 22:55:52
*/


-- ----------------------------
-- Table structure for future_code
-- ----------------------------
IF EXISTS (SELECT * FROM sys.all_objects WHERE object_id = OBJECT_ID(N'[dbo].[future_code]') AND type IN ('U'))
	DROP TABLE [dbo].[future_code]
GO

CREATE TABLE [dbo].[future_code] (
  [ID] int  IDENTITY(1,1) NOT FOR REPLICATION NOT NULL,
  [code] varchar(20) COLLATE SQL_Latin1_General_CP1_CI_AS  NULL,
  [name] varchar(20) COLLATE SQL_Latin1_General_CP1_CI_AS  NULL,
  [de_listed_date] datetime  NULL,
  [exchange] varchar(20) COLLATE SQL_Latin1_General_CP1_CI_AS  NULL,
  [market_tplus] varchar(20) COLLATE SQL_Latin1_General_CP1_CI_AS  NULL,
  [margin_rate] real  NULL,
  [symbol] nvarchar(20) COLLATE SQL_Latin1_General_CP1_CI_AS  NULL,
  [order_book_id] varchar(20) COLLATE SQL_Latin1_General_CP1_CI_AS  NULL,
  [underlying_symbol] varchar(20) COLLATE SQL_Latin1_General_CP1_CI_AS  NULL,
  [contract_multiplier] real  NULL,
  [listed_date] datetime  NULL,
  [trading_hours] varchar(100) COLLATE SQL_Latin1_General_CP1_CI_AS  NULL,
  [type] varchar(20) COLLATE SQL_Latin1_General_CP1_CI_AS  NULL,
  [maturity_date] datetime  NULL,
  [main_date] datetime  NULL,
  [used_date] datetime  NULL,
  [used_volume] real  NULL,
  [de_used_date] datetime  NULL
)
GO

ALTER TABLE [dbo].[future_code] SET (LOCK_ESCALATION = TABLE)
GO

