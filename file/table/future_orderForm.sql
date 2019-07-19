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

 Date: 15/11/2018 20:07:50
*/


-- ----------------------------
-- Table structure for sh50_orderForm
-- ----------------------------
IF EXISTS (SELECT * FROM sys.all_objects WHERE object_id = OBJECT_ID(N'[dbo].[sh50_orderForm]') AND type IN ('U'))
	DROP TABLE [dbo].[sh50_orderForm]
GO

CREATE TABLE [dbo].[sh50_orderForm] (
  [ID] int  IDENTITY(1,1) NOT FOR REPLICATION NOT NULL,
  [code] varchar(20) COLLATE SQL_Latin1_General_CP1_CI_AS  NULL,
  [createdate] datetime  NULL,
  [price] real  NULL,
  [vol] real  NULL,
  [mode] int  NULL,
  [isBuy] int  NULL,
  [pos] int NULL,
  [ownerPrice] real NULL,
  [fee] real  NULL,
  [amount] real  NULL,
  [income] real  NULL,
  [batchid] varchar(100) COLLATE SQL_Latin1_General_CP1_CI_AS  NULL,
  [uid] varchar(200) COLLATE SQL_Latin1_General_CP1_CI_AS  NULL,
  [interval] int  NULL,
  [orderID] int  NULL,
  [method] varchar(10) COLLATE SQL_Latin1_General_CP1_CI_AS  NULL,
  [status] int  NULL,
  [ini_price] real  NULL,
  [ini_hands] real  NULL
 )
GO

ALTER TABLE [dbo].[sh50_orderForm] SET (LOCK_ESCALATION = TABLE)
GO

