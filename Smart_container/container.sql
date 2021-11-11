/*
Navicat MySQL Data Transfer

Source Server         : localhost_3306
Source Server Version : 50721
Source Host           : localhost:3306
Source Database       : container

Target Server Type    : MYSQL
Target Server Version : 50721
File Encoding         : 65001

Date: 2021-11-02 17:20:43
*/

SET FOREIGN_KEY_CHECKS=0;

-- ----------------------------
-- Table structure for `t_container`
-- ----------------------------
DROP TABLE IF EXISTS `t_container`;
CREATE TABLE `t_container` (
  `number` int(11) NOT NULL,
  `openid` varchar(255) DEFAULT NULL,
  `container_name` varchar(255) DEFAULT NULL,
  `container_price` varchar(255) DEFAULT NULL,
  `picture_address` varchar(255) DEFAULT NULL,
  `stock` int(11) DEFAULT NULL,
  PRIMARY KEY (`number`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- ----------------------------
-- Records of t_container
-- ----------------------------
INSERT INTO `t_container` VALUES ('1', null, 'airpods2', '998', 'gallery/airpods2.jpg', null);
INSERT INTO `t_container` VALUES ('2', null, 'HUAWEI_P30_PRO', '2020', 'gallery/HUAWEI_P30_PRO.jpg', null);
INSERT INTO `t_container` VALUES ('3', null, 'HUAWEI_WATCH_3_Pro', '1388', 'gallery/HUAWEI_WATCH_3_Pro.jpg', null);
INSERT INTO `t_container` VALUES ('4', null, 'iphone_13', '6388', 'gallery/iphone_13.jpg', null);
INSERT INTO `t_container` VALUES ('5', null, 'iQOO_7', '2658', 'gallery/iQOO_7.jpg', null);
INSERT INTO `t_container` VALUES ('6', null, 'Meco果汁茶桃桃红柚', '48', 'gallery/Mecoguozhichataotaohongyou.jpg', null);
INSERT INTO `t_container` VALUES ('7', null, 'Meco果汁茶泰式青柠', '48', 'gallery/Mecoguozhichataishiqingning.jpg', null);
INSERT INTO `t_container` VALUES ('8', null, 'Redmi_K40', '2149', 'gallery/Redmi_K40.jpg', null);
INSERT INTO `t_container` VALUES ('9', null, 'VIVO_x27', '1368', 'gallery/VIVO_x27.jpg', null);
INSERT INTO `t_container` VALUES ('10', null, 'VOSS矿泉水', '45', 'gallery/VOSSkuangquanshui.jpg', null);
INSERT INTO `t_container` VALUES ('11', null, 'Xiaomi_Civi', '2599', 'gallery/Xiaomi_Civi.jpg', null);
INSERT INTO `t_container` VALUES ('12', null, '七喜330ml', '3', 'gallery/qixi330ml.jpg', null);
INSERT INTO `t_container` VALUES ('13', null, '三得利乌龙茶', '4', 'gallery/sandeliwulongcha.jpg', null);
INSERT INTO `t_container` VALUES ('14', null, '东方树叶乌龙茶', '4', 'gallery/dongfangshuyewulongcha.jpg', null);
INSERT INTO `t_container` VALUES ('15', null, '东方树叶红茶', '4', 'gallery/dongfangshuyehongcha.jpg', null);
INSERT INTO `t_container` VALUES ('16', null, '东方树叶茉莉花茶', '4', 'gallery/dongfangshuyemolihuacha.jpg', null);
INSERT INTO `t_container` VALUES ('17', null, '东鹏特饮瓶装', '4', 'gallery/dongpengteyinpingzhuang.jpg', null);
INSERT INTO `t_container` VALUES ('18', null, '东鹏特饮组合装', '50', 'gallery/dongpengteyinzuhezhuang.jpg', null);
INSERT INTO `t_container` VALUES ('19', null, '乳酸菌600亿_2', '15', 'gallery/rusuanjun600yi_2.jpg', null);
INSERT INTO `t_container` VALUES ('20', null, '乳酸菌600亿_3', '15', 'gallery/rusuanjun600yi_3.jpg', null);
INSERT INTO `t_container` VALUES ('21', null, '乳酸菌600亿原味', '15', 'gallery/rusuanjun600yiyuanwei.jpg', null);
INSERT INTO `t_container` VALUES ('22', null, '乳酸菌600亿芒果', '15', 'gallery/rusuanjun600yimangguo.jpg', null);
INSERT INTO `t_container` VALUES ('23', null, '乳酸菌600亿芦荟', '15', 'gallery/rusuanjun600yiluhui.jpg', null);
INSERT INTO `t_container` VALUES ('24', null, '乳酸菌600亿草莓', '15', 'gallery/rusuanjun600yicaomei.jpg', null);
INSERT INTO `t_container` VALUES ('25', null, '乳酸菌600亿西瓜', '15', 'gallery/rusuanjun600yixigua.jpg', null);
INSERT INTO `t_container` VALUES ('26', null, '伊利安慕希瓶装原味230g', '78', 'gallery/yilianmuxipingzhuangyuanwei230g.jpg', null);
INSERT INTO `t_container` VALUES ('27', null, '伊利安慕希草莓味酸奶205g', '45', 'gallery/yilianmuxicaomeiweisuannai205g.jpg', null);
INSERT INTO `t_container` VALUES ('28', null, '伊利安慕希草莓燕麦味酸奶200g', '58', 'gallery/yilianmuxicaomeiyanmaiweisuannai200g.jpg', null);
INSERT INTO `t_container` VALUES ('29', null, '伊利安慕希高端原味230ml', '78', 'gallery/yilianmuxigaoduanyuanwei230ml.jpg', null);
INSERT INTO `t_container` VALUES ('30', null, '伊利安慕希高端橙凤梨味230ml', '70.9', 'gallery/yilianmuxigaoduanchengfengliwei230ml.jpg', null);
INSERT INTO `t_container` VALUES ('31', null, '伊利纯牛奶250ml', '60.8', 'gallery/yilichunniunai250ml.jpg', null);
INSERT INTO `t_container` VALUES ('32', null, '优倍', '18.27', 'gallery/youbei.jpg', null);
INSERT INTO `t_container` VALUES ('33', null, '优酪乳健康大麦180g_4麦香原味', '24.8', 'gallery/youlaorujiankangdamai180g_4maixiangyuanwei.jpg', null);
INSERT INTO `t_container` VALUES ('34', null, '优酪乳健康大麦180g草莓味', '24.8', 'gallery/youlaorujiankangdamai180gcaomeiwei.jpg', null);
INSERT INTO `t_container` VALUES ('35', null, '优酪乳健康大麦180g麦香原味', '24.8', 'gallery/youlaorujiankangdamai180gmaixiangyuanwei.jpg', null);
INSERT INTO `t_container` VALUES ('36', null, '优酪乳唯果粒230g芒果黄桃', '106.9', 'gallery/youlaoruweiguoli230gmangguohuangtao.jpg', null);
INSERT INTO `t_container` VALUES ('37', null, '优酪乳唯果粒230g芦荟', '106.9', 'gallery/youlaoruweiguoli230gluhui.jpg', null);
INSERT INTO `t_container` VALUES ('38', null, '优酪乳唯果粒230g草莓', '106.9', 'gallery/youlaoruweiguoli230gcaomei.jpg', null);
INSERT INTO `t_container` VALUES ('39', null, '优酪乳四季鲜选180g_4芦荟', '85', 'gallery/youlaorusijixianxuan180g_4luhui.jpg', null);
INSERT INTO `t_container` VALUES ('40', null, '优酪乳四季鲜选180g_4黄桃', '85', 'gallery/youlaorusijixianxuan180g_4huangtao.jpg', null);
INSERT INTO `t_container` VALUES ('41', null, '优酪乳四季鲜选180g芦荟', '85', 'gallery/youlaorusijixianxuan180gluhui.jpg', null);
INSERT INTO `t_container` VALUES ('42', null, '优酪乳四季鲜选180g黄桃', '85', 'gallery/youlaorusijixianxuan180ghuangtao.jpg', null);
INSERT INTO `t_container` VALUES ('43', null, '优酪乳慢一点100g_8原味', '28.52', 'gallery/youlaorumanyidian100g_8yuanwei.jpg', null);
INSERT INTO `t_container` VALUES ('44', null, '优酪乳旅行优格220g丹麦芝士味', '5', 'gallery/youlaorulvxingyouge220gdanmaizhishiwei.jpg', null);
INSERT INTO `t_container` VALUES ('45', null, '优酪乳旅行优格220g保加利亚玫瑰味', '5', 'gallery/youlaorulvxingyouge220gbaojialiyameiguiwei.jpg', null);
INSERT INTO `t_container` VALUES ('46', null, '优酪乳简单点100g_8原味', '5', 'gallery/youlaorujiandandian100g_8yuanwei.jpg', null);
INSERT INTO `t_container` VALUES ('47', null, '优酪乳简单点230g半糖', '5', 'gallery/youlaorujiandandian230gbantang.jpg', null);
INSERT INTO `t_container` VALUES ('48', null, '优酪乳简单点原味', '6', 'gallery/youlaorujiandandianyuanwei.jpg', null);
INSERT INTO `t_container` VALUES ('49', null, '优酪乳简单点烤酸奶', '7', 'gallery/youlaorujiandandiankaosuannai.jpg', null);
INSERT INTO `t_container` VALUES ('50', null, '优酪乳顺畅点230g原味', '5', 'gallery/youlaorushunchangdian230gyuanwei.jpg', null);
INSERT INTO `t_container` VALUES ('51', null, '伴侣酱油1L', '25', 'gallery/banlvjiangyou1L.jpg', null);
INSERT INTO `t_container` VALUES ('52', null, '伴侣酱油2L', '40', 'gallery/banlvjiangyou2L.jpg', null);
INSERT INTO `t_container` VALUES ('53', null, '元气森林乳茶咖啡拿铁', '90', 'gallery/yuanqisenlinruchakafeinatie.jpg', null);
INSERT INTO `t_container` VALUES ('54', null, '元气森林乳茶浓香原味', '90', 'gallery/yuanqisenlinruchanongxiangyuanwei.jpg', null);
INSERT INTO `t_container` VALUES ('55', null, '元气森林乳茶茉香奶绿', '90', 'gallery/yuanqisenlinruchamoxiangnailv.jpg', null);
INSERT INTO `t_container` VALUES ('56', null, '元气森林乳酸菌苏打气泡水', '129.9', 'gallery/yuanqisenlinrusuanjunsudaqipaoshui.jpg', null);
INSERT INTO `t_container` VALUES ('57', null, '元气森林无糖卡曼橘味苏打气泡水', '129.9', 'gallery/yuanqisenlinwutangqiamanjuweisudaqipaoshui.jpg', null);
INSERT INTO `t_container` VALUES ('58', null, '元气森林无糖白桃味苏打气泡水', '129.9', 'gallery/yuanqisenlinwutangbaitaoweisudaqipaoshui.jpg', null);
INSERT INTO `t_container` VALUES ('59', null, '元气森林樱花葡萄苏打气泡水', '129.9', 'gallery/yuanqisenlinyinghuaputaosudaqipaoshui.jpg', null);
INSERT INTO `t_container` VALUES ('60', null, '元气森林燃茶无糖桃香乌龙茶', '94.9', 'gallery/yuanqisenlinranchawutangtaoxiangwulongcha.jpg', null);
INSERT INTO `t_container` VALUES ('61', null, '元气森林燃茶无糖醇香乌龙茶', '94.9', 'gallery/yuanqisenlinranchawutangchunxiangwulongcha.jpg', null);
INSERT INTO `t_container` VALUES ('62', null, '元气森林燃茶玄米乌龙茶', '94.9', 'gallery/yuanqisenlinranchaxuanmiwulongcha.jpg', null);
INSERT INTO `t_container` VALUES ('63', null, '元气森林白桃味苏打气泡水', '86.9', 'gallery/yuanqisenlinbaitaoweisudaqipaoshui.jpg', null);
INSERT INTO `t_container` VALUES ('64', null, '元气森林酸梅汁苏打气泡水', '86.9', 'gallery/yuanqisenlinsuanmeizhisudaqipaoshui.jpg', null);
INSERT INTO `t_container` VALUES ('65', null, '元气森林青瓜味苏打气泡水', '86.9', 'gallery/yuanqisenlinqingguaweisudaqipaoshui.jpg', null);
INSERT INTO `t_container` VALUES ('66', null, '光明藜麦', '5', 'gallery/guangminglimai.jpg', null);
INSERT INTO `t_container` VALUES ('67', null, '光明鲜牛奶', '5', 'gallery/guangmingxianniunai.jpg', null);
INSERT INTO `t_container` VALUES ('68', null, '养乐多', '6', 'gallery/yangleduo.jpg', null);
INSERT INTO `t_container` VALUES ('69', null, '养乐多组合装', '52.8', 'gallery/yangleduozuhezhuang.jpg', null);
INSERT INTO `t_container` VALUES ('70', null, '农夫山泉17.5', '69.9', 'gallery/nongfushanquan17.5.jpg', null);
INSERT INTO `t_container` VALUES ('71', null, '农夫山泉NFC', '38', 'gallery/nongfushanquanNFC.jpg', null);
INSERT INTO `t_container` VALUES ('72', null, '农夫山泉维他命水乳酸菌风味', '48', 'gallery/nongfushanquanweitamingshuirusuanjunfengwei.jpg', null);
INSERT INTO `t_container` VALUES ('73', null, '农夫山泉维他命水柑橘风味', '48', 'gallery/nongfushanquanweitamingshuiganjufengwei.jpg', null);
INSERT INTO `t_container` VALUES ('74', null, '农夫山泉维他命水柠檬风味', '48', 'gallery/nongfushanquanweitamingshuiningmengfengwei.jpg', null);
INSERT INTO `t_container` VALUES ('75', null, '农夫山泉维他命水热带水果风味', '48', 'gallery/nongfushanquanweitamingshuiredaishuiguofengwei.jpg', null);
INSERT INTO `t_container` VALUES ('76', null, '农夫山泉维他命水石榴蓝莓风味', '48', 'gallery/nongfushanquanweitamingshuishiliulanmeifengwei.jpg', null);
INSERT INTO `t_container` VALUES ('77', null, '农夫山泉维他命水蓝莓树莓味', '48', 'gallery/nongfushanquanweitamingshuilanmeishumeiwei.jpg', null);
INSERT INTO `t_container` VALUES ('78', null, '农夫山泉茶π柚子绿茶', '64.9', 'gallery/nongfushanquanchaπyouzilvcha.jpg', null);
INSERT INTO `t_container` VALUES ('79', null, '农夫山泉茶π柠檬红茶', '64.9', 'gallery/nongfushanquanchaπningmenghongcha.jpg', null);
INSERT INTO `t_container` VALUES ('80', null, '农夫山泉茶π玫瑰荔枝红茶', '64.9', 'gallery/nongfushanquanchaπmeiguilizhihongcha.jpg', null);
INSERT INTO `t_container` VALUES ('81', null, '农夫山泉茶π蜜桃乌龙茶', '64.9', 'gallery/nongfushanquanchaπmitaowulongcha.jpg', null);
INSERT INTO `t_container` VALUES ('82', null, '农夫山泉茶π西柚茉莉花茶', '64.9', 'gallery/nongfushanquanchaπxiyoumolihuacha.jpg', null);
INSERT INTO `t_container` VALUES ('83', null, '冠益乳', '34.8', 'gallery/guanyiru.jpg', null);
INSERT INTO `t_container` VALUES ('84', null, '冰露矿泉水组合装', '9.8', 'gallery/binglukuangquanshuizuhezhuang.jpg', null);
INSERT INTO `t_container` VALUES ('85', null, '冰露纯悦矿泉水', '29.1', 'gallery/bingluchunyuekuangquanshui.jpg', null);
INSERT INTO `t_container` VALUES ('86', null, '凯旋1664blanc啤酒瓶装', '7', 'gallery/kaixuan1664blancpijiupingzhuang.jpg', null);
INSERT INTO `t_container` VALUES ('87', null, '凯旋1664blanc啤酒罐啤', '6', 'gallery/kaixuan1664blancpijiuguanpi.jpg', null);
INSERT INTO `t_container` VALUES ('88', null, '午后红茶柠檬味', '4', 'gallery/wuhouhongchaningmengwei.jpg', null);
INSERT INTO `t_container` VALUES ('89', null, '华为Mate_40', '5388', 'gallery/huaweiMate_40.jpg', null);
INSERT INTO `t_container` VALUES ('90', null, '华为P50_Pro', '5538', 'gallery/huaweiP50_Pro.jpg', null);
INSERT INTO `t_container` VALUES ('91', null, '小米11', '4098', 'gallery/xiaomi11.jpg', null);
INSERT INTO `t_container` VALUES ('92', null, '尖叫多肽型', '121.5', 'gallery/jianjiaoduotaixing.jpg', null);
INSERT INTO `t_container` VALUES ('93', null, '康师傅冰糖乌龙茶', '35', 'gallery/kangshifubingtangwulongcha.jpg', null);
INSERT INTO `t_container` VALUES ('94', null, '康师傅冰糖柠檬', '35', 'gallery/kangshifubingtangningmeng.jpg', null);
INSERT INTO `t_container` VALUES ('95', null, '康师傅冰糖雪梨', '35', 'gallery/kangshifubingtangxueli.jpg', null);
INSERT INTO `t_container` VALUES ('96', null, '康师傅冰红茶', '35', 'gallery/kangshifubinghongcha.jpg', null);
INSERT INTO `t_container` VALUES ('97', null, '康师傅冰绿茶', '35', 'gallery/kangshifubinglvcha.jpg', null);
INSERT INTO `t_container` VALUES ('98', null, '康师傅绿茶', '35', 'gallery/kangshifulvcha.jpg', null);
INSERT INTO `t_container` VALUES ('99', null, '康师傅茉莉柚茶', '35', 'gallery/kangshifumoliyoucha.jpg', null);
INSERT INTO `t_container` VALUES ('100', null, '康师傅茉莉清茶', '35', 'gallery/kangshifumoliqingcha.jpg', null);
INSERT INTO `t_container` VALUES ('101', null, '康师傅蜂蜜柚子', '35', 'gallery/kangshifufengmiyouzi.jpg', null);
INSERT INTO `t_container` VALUES ('102', null, '得力卡通长尾票夹', '7.8', 'gallery/deliqiatongchangweipiaojia.jpg', null);
INSERT INTO `t_container` VALUES ('103', null, '得力文件夹', '4.8', 'gallery/deliwenjianjia.jpg', null);
INSERT INTO `t_container` VALUES ('104', null, '旺仔牛奶', '70', 'gallery/wangziniunai.jpg', null);
INSERT INTO `t_container` VALUES ('105', null, '明治醇壹_优漾', '8', 'gallery/mingzhichunyi_youyang.jpg', null);
INSERT INTO `t_container` VALUES ('106', null, '星巴克250ml咖啡拿铁', '127', 'gallery/xingbake250mlkafeinatie.jpg', null);
INSERT INTO `t_container` VALUES ('107', null, '星巴克250ml抹茶拿铁', '127', 'gallery/xingbake250mlmochanatie.jpg', null);
INSERT INTO `t_container` VALUES ('108', null, '星巴克250ml香草拿铁', '127', 'gallery/xingbake250mlxiangcaonatie.jpg', null);
INSERT INTO `t_container` VALUES ('109', null, '椰树牌椰汁', '89', 'gallery/yeshupaiyezhi.jpg', null);
INSERT INTO `t_container` VALUES ('110', null, '欣和寿司醋245ML', '12', 'gallery/xinheshousicu245ML.jpg', null);
INSERT INTO `t_container` VALUES ('111', null, '每日C果蔬汁300ml树莓红甜菜', '60', 'gallery/meiriCguoshuzhi300mlshumeihongtiancai.jpg', null);
INSERT INTO `t_container` VALUES ('112', null, '每日C果蔬汁300ml百香果南瓜', '60', 'gallery/meiriCguoshuzhi300mlbaixiangguonangua.jpg', null);
INSERT INTO `t_container` VALUES ('113', null, '每日C果蔬汁300ml金桔羽衣甘蓝', '60', 'gallery/meiriCguoshuzhi300mljinjieyuyiganlan.jpg', null);
INSERT INTO `t_container` VALUES ('114', null, '每日C橙汁300ml', '60', 'gallery/meiriCchengzhi300ml.jpg', null);
INSERT INTO `t_container` VALUES ('115', null, '每日C纯果汁果纤橙', '60.9', 'gallery/meiriCchunguozhiguoxiancheng.jpg', null);
INSERT INTO `t_container` VALUES ('116', null, '每日C纯果汁桃汁', '60.9', 'gallery/meiriCchunguozhitaozhi.jpg', null);
INSERT INTO `t_container` VALUES ('117', null, '每日C纯果汁橙汁', '60.9', 'gallery/meiriCchunguozhichengzhi.jpg', null);
INSERT INTO `t_container` VALUES ('118', null, '每日C纯果汁胡萝卜汁', '60.9', 'gallery/meiriCchunguozhihuluobuzhi.jpg', null);
INSERT INTO `t_container` VALUES ('119', null, '每日C纯果汁芒果', '60.9', 'gallery/meiriCchunguozhimangguo.jpg', null);
INSERT INTO `t_container` VALUES ('120', null, '每日C纯果汁苹果', '60.9', 'gallery/meiriCchunguozhipingguo.jpg', null);
INSERT INTO `t_container` VALUES ('121', null, '每日C纯果汁葡萄', '60.9', 'gallery/meiriCchunguozhiputao.jpg', null);
INSERT INTO `t_container` VALUES ('122', null, '每日C纯果汁葡萄柚', '60.9', 'gallery/meiriCchunguozhiputaoyou.jpg', null);
INSERT INTO `t_container` VALUES ('123', null, '每益添', '29.8', 'gallery/meiyitian.jpg', null);
INSERT INTO `t_container` VALUES ('124', null, '水动乐桃味600ml', '54.5', 'gallery/shuidongletaowei600ml.jpg', null);
INSERT INTO `t_container` VALUES ('125', null, '法国原装进口巴黎水perrier原味', '77', 'gallery/faguoyuanzhuangjinkoubalishuiperrieryuanwei.jpg', null);
INSERT INTO `t_container` VALUES ('126', null, '法国原装进口巴黎水perrier青柠', '77', 'gallery/faguoyuanzhuangjinkoubalishuiperrierqingning.jpg', null);
INSERT INTO `t_container` VALUES ('127', null, '百事可乐无糖罐装', '69', 'gallery/baishikelewutangguanzhuang.jpg', null);
INSERT INTO `t_container` VALUES ('128', null, '百事可乐罐装塑封', '69', 'gallery/baishikeleguanzhuangsufeng.jpg', null);
INSERT INTO `t_container` VALUES ('129', null, '百威啤酒红色铝罐', '69', 'gallery/baiweipijiuhongselvguan.jpg', null);
INSERT INTO `t_container` VALUES ('130', null, '百威啤酒金尊550罐啤', '69', 'gallery/baiweipijiujinzun550guanpi.jpg', null);
INSERT INTO `t_container` VALUES ('131', null, '百岁山500ml', '11', 'gallery/baisuishan500ml.jpg', null);
INSERT INTO `t_container` VALUES ('132', null, '签字笔', '10.2', 'gallery/qianzibi.jpg', null);
INSERT INTO `t_container` VALUES ('133', null, '统一绿茶', '37.9', 'gallery/tongyilvcha.jpg', null);
INSERT INTO `t_container` VALUES ('134', null, '美年达柠檬味', '12', 'gallery/meiniandaningmengwei.jpg', null);
INSERT INTO `t_container` VALUES ('135', null, '美年达橙味汽水500ml', '29.9', 'gallery/meiniandachengweiqishui500ml.jpg', null);
INSERT INTO `t_container` VALUES ('136', null, '美年达橙味罐装', '45.9', 'gallery/meiniandachengweiguanzhuang.jpg', null);
INSERT INTO `t_container` VALUES ('137', null, '美年达青苹果味汽水500ml', '29.9', 'gallery/meiniandaqingpingguoweiqishui500ml.jpg', null);
INSERT INTO `t_container` VALUES ('138', null, '芬达椰子味瓶装', '5', 'gallery/fendayeziweipingzhuang.jpg', null);
INSERT INTO `t_container` VALUES ('139', null, '芬达苹果味瓶装', '5', 'gallery/fendapingguoweipingzhuang.jpg', null);
INSERT INTO `t_container` VALUES ('140', null, '芬达苹果味罐装', '5', 'gallery/fendapingguoweiguanzhuang.jpg', null);
INSERT INTO `t_container` VALUES ('141', null, '芬达菠萝味瓶装', '5', 'gallery/fendaboluoweipingzhuang.jpg', null);
INSERT INTO `t_container` VALUES ('142', null, '芬达葡萄味瓶装', '5', 'gallery/fendaputaoweipingzhuang.jpg', null);
INSERT INTO `t_container` VALUES ('143', null, '芬达葡萄味罐装', '5', 'gallery/fendaputaoweiguanzhuang.jpg', null);
INSERT INTO `t_container` VALUES ('144', null, '芬达蜜桃味瓶装', '5', 'gallery/fendamitaoweipingzhuang.jpg', null);
INSERT INTO `t_container` VALUES ('145', null, '芬达蜜桃味罐装', '5', 'gallery/fendamitaoweiguanzhuang.jpg', null);
INSERT INTO `t_container` VALUES ('146', null, '芬达西瓜味瓶装', '5', 'gallery/fendaxiguaweipingzhuang.jpg', null);
INSERT INTO `t_container` VALUES ('147', null, '芬达西瓜味罐装', '5', 'gallery/fendaxiguaweiguanzhuang.jpg', null);
INSERT INTO `t_container` VALUES ('148', null, '芬达零卡橙味瓶装', '69', 'gallery/fendalingqiachengweipingzhuang.jpg', null);
INSERT INTO `t_container` VALUES ('149', null, '茶π柚子绿茶', '64.9', 'gallery/chaπyouzilvcha.jpg', null);
INSERT INTO `t_container` VALUES ('150', null, '茶π柠檬红茶', '64.9', 'gallery/chaπningmenghongcha.jpg', null);
INSERT INTO `t_container` VALUES ('151', null, '茶π玫瑰荔枝红茶', '64.9', 'gallery/chaπmeiguilizhihongcha.jpg', null);
INSERT INTO `t_container` VALUES ('152', null, '茶π蜜桃乌龙茶', '64.9', 'gallery/chaπmitaowulongcha.jpg', null);
INSERT INTO `t_container` VALUES ('153', null, '茶π西柚茉莉花茶', '64.9', 'gallery/chaπxiyoumolihuacha.jpg', null);
INSERT INTO `t_container` VALUES ('154', null, '蒙牛优益c', '30.8', 'gallery/mengniuyouyic.jpg', null);
INSERT INTO `t_container` VALUES ('155', null, '蒙牛纯甄瓶装红西柚味酸奶230g', '80', 'gallery/mengniuchunzhenpingzhuanghongxiyouweisuannai230g.jpg', null);
INSERT INTO `t_container` VALUES ('156', null, '谷物牛乳饮品300g燕麦谷粒', '4', 'gallery/guwuniuruyinpin300gyanmaiguli.jpg', null);
INSERT INTO `t_container` VALUES ('157', null, '谷物牛乳饮品300g红豆紫米', '4', 'gallery/guwuniuruyinpin300ghongdouzimi.jpg', null);
INSERT INTO `t_container` VALUES ('158', null, '谷物牛乳饮品300g藜麦玉米', '4', 'gallery/guwuniuruyinpin300glimaiyumi.jpg', null);
INSERT INTO `t_container` VALUES ('159', null, '谷物牛乳饮品950g燕麦谷粒', '9', 'gallery/guwuniuruyinpin950gyanmaiguli.jpg', null);
INSERT INTO `t_container` VALUES ('160', null, '谷物牛乳饮品950g红豆紫米', '9', 'gallery/guwuniuruyinpin950ghongdouzimi.jpg', null);
INSERT INTO `t_container` VALUES ('161', null, '豪格登啤酒', '8', 'gallery/haogedengpijiu.jpg', null);
INSERT INTO `t_container` VALUES ('162', null, '贝瑞斯塔barista', '39', 'gallery/beiruisitabarista.jpg', null);
INSERT INTO `t_container` VALUES ('163', null, '贝纳颂名地臻选250ml西达摩', '79', 'gallery/beinasongmingdizhenxuan250mlxidamo.jpg', null);
INSERT INTO `t_container` VALUES ('164', null, '贝纳颂咖啡拿铁', '8.8', 'gallery/beinasongkafeinatie.jpg', null);
INSERT INTO `t_container` VALUES ('165', null, '贝纳颂经典系列250ml拿铁', '59.9', 'gallery/beinasongjingdianxilie250mlnatie.jpg', null);
INSERT INTO `t_container` VALUES ('166', null, '贝纳颂经典系列250ml摩卡', '59.9', 'gallery/beinasongjingdianxilie250mlmoqia.jpg', null);
INSERT INTO `t_container` VALUES ('167', null, '贝纳颂经典系列250ml蓝山', '59.9', 'gallery/beinasongjingdianxilie250mllanshan.jpg', null);
INSERT INTO `t_container` VALUES ('168', null, '阿萨姆奶茶', '39.9', 'gallery/asamunaicha.jpg', null);
INSERT INTO `t_container` VALUES ('169', null, '雀巢美极鲜', '10.8', 'gallery/quechaomeijixian.jpg', null);
INSERT INTO `t_container` VALUES ('170', null, '雪碧罐装', '58.8', 'gallery/xuebiguanzhuang.jpg', null);
INSERT INTO `t_container` VALUES ('171', null, '雪花8度勇闯天涯500ml瓶装', '5', 'gallery/xuehua8duyongchuangtianya500mlpingzhuang.jpg', null);
INSERT INTO `t_container` VALUES ('172', null, '雪花8度勇闯天涯500ml罐啤', '5', 'gallery/xuehua8duyongchuangtianya500mlguanpi.jpg', null);
INSERT INTO `t_container` VALUES ('173', null, '雪花8度勇闯天涯罐啤_6组合装', '30', 'gallery/xuehua8duyongchuangtianyaguanpi_6zuhezhuang.jpg', null);
INSERT INTO `t_container` VALUES ('174', null, '雪花8度清爽', '59', 'gallery/xuehua8duqingshuang.jpg', null);
INSERT INTO `t_container` VALUES ('175', null, '雪花9度勇闯天涯500ml瓶装', '6', 'gallery/xuehua9duyongchuangtianya500mlpingzhuang.jpg', null);
INSERT INTO `t_container` VALUES ('176', null, '雪花冰酷330ml箱装', '50', 'gallery/xuehuabingku330mlxiangzhuang.jpg', null);
INSERT INTO `t_container` VALUES ('177', null, '雪花冰酷9度罐啤', '25.9', 'gallery/xuehuabingku9duguanpi.jpg', null);
INSERT INTO `t_container` VALUES ('178', null, '雪花勇闯天涯superX', '84', 'gallery/xuehuayongchuangtianyasuperX.jpg', null);
INSERT INTO `t_container` VALUES ('179', null, '雪花清爽8度330_6罐啤组合装', '30', 'gallery/xuehuaqingshuang8du330_6guanpizuhezhuang.jpg', null);
INSERT INTO `t_container` VALUES ('180', null, '雪花清爽8度箱装', '59', 'gallery/xuehuaqingshuang8duxiangzhuang.jpg', null);
INSERT INTO `t_container` VALUES ('181', null, '雪花精制9度500_6罐啤组合装', '30', 'gallery/xuehuajingzhi9du500_6guanpizuhezhuang.jpg', null);
INSERT INTO `t_container` VALUES ('182', null, '雪花纯生500ml瓶装', '33', 'gallery/xuehuachunsheng500mlpingzhuang.jpg', null);
INSERT INTO `t_container` VALUES ('183', null, '雪花纯生500ml组合装', '30', 'gallery/xuehuachunsheng500mlzuhezhuang.jpg', null);
INSERT INTO `t_container` VALUES ('184', null, '雪花纯生罐啤', '86', 'gallery/xuehuachunshengguanpi.jpg', null);
INSERT INTO `t_container` VALUES ('185', null, '雪花脸谱花旦系列8度500罐啤', '180', 'gallery/xuehualianpuhuadanxilie8du500guanpi.jpg', null);
INSERT INTO `t_container` VALUES ('186', null, '青岛啤酒全麦白啤500罐啤', '99', 'gallery/qingdaopijiuquanmaibaipi500guanpi.jpg', null);
INSERT INTO `t_container` VALUES ('187', null, '青岛啤酒奥古特500ml', '178', 'gallery/qingdaopijiuaogute500ml.jpg', null);
INSERT INTO `t_container` VALUES ('188', null, '青岛啤酒小棕金296瓶装', '138', 'gallery/qingdaopijiuxiaozongjin296pingzhuang.jpg', null);
INSERT INTO `t_container` VALUES ('189', null, '青岛啤酒淡爽8度330罐啤', '89', 'gallery/qingdaopijiudanshuang8du330guanpi.jpg', null);
INSERT INTO `t_container` VALUES ('190', null, '青岛啤酒清醇330ml', '89', 'gallery/qingdaopijiuqingchun330ml.jpg', null);
INSERT INTO `t_container` VALUES ('191', null, '青岛啤酒纯生500ml罐啤', '119', 'gallery/qingdaopijiuchunsheng500mlguanpi.jpg', null);
INSERT INTO `t_container` VALUES ('192', null, '青岛啤酒纯生600ml瓶装', '68', 'gallery/qingdaopijiuchunsheng600mlpingzhuang.jpg', null);
INSERT INTO `t_container` VALUES ('193', null, '青岛啤酒经典10度500罐啤', '89', 'gallery/qingdaopijiujingdian10du500guanpi.jpg', null);
INSERT INTO `t_container` VALUES ('194', null, '青岛啤酒经典11度330罐啤', '86', 'gallery/qingdaopijiujingdian11du330guanpi.jpg', null);
INSERT INTO `t_container` VALUES ('195', null, '青岛啤酒经典11度罐啤牛卡纸组合装', '40', 'gallery/qingdaopijiujingdian11duguanpiniuqiazhizuhezhuang.jpg', null);
INSERT INTO `t_container` VALUES ('196', null, '青岛啤酒经典（1903）10度330_6罐啤', '29.9', 'gallery/qingdaopijiujingdian（1903）10du330_6guanpi.jpg', null);
INSERT INTO `t_container` VALUES ('197', null, '青岛啤酒鸿运当头355瓶装', '28', 'gallery/qingdaopijiuhongyundangtou355pingzhuang.jpg', null);
INSERT INTO `t_container` VALUES ('198', null, '青岛啤酒黑啤酒500罐啤', '49.9', 'gallery/qingdaopijiuheipijiu500guanpi.jpg', null);
INSERT INTO `t_container` VALUES ('199', null, '魅族18', '3098', 'gallery/meizu18.jpg', null);

-- ----------------------------
-- Table structure for `t_user`
-- ----------------------------
DROP TABLE IF EXISTS `t_user`;
CREATE TABLE `t_user` (
  `openid` varchar(255) NOT NULL,
  `nickname` varchar(255) DEFAULT NULL,
  `session_key` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`openid`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

-- ----------------------------
-- Records of t_user
-- ----------------------------
