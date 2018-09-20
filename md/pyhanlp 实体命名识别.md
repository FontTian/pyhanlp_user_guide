
## pyhanlp中的命名实体识别

对于分词而言，命名实体识别是一项非常重要的功能，当然发现新词同样重要（这部分内容被我放在之后的“关键词、短语提取与自动摘要、新词识别”与再之后的案例中了。

首先是一个简单的例子，展示一下命名实体识别的效果。之后是正式内容：

## 简单的展示例子


```python
from pyhanlp import *
"""
HanLP开启命名实体识别

"""

# 音译人名示例
CRFnewSegment = HanLP.newSegment("crf")
term_list = CRFnewSegment.seg("译智社的田丰要说的是这只是一个hanlp命名实体识别的例子")
print(term_list)


print("\n========== 命名实体开启与关闭对比试验 ==========\n")
sentences =[
    "北川景子参演了林诣彬导演的《速度与激情3》",
    "林志玲亮相网友:确定不是波多野结衣？",
    "龟山千广和近藤公园在龟山公园里喝酒赏花",
]
# 通过HanLP 进行全局设置,但是部分分词器本身可能不支持某项功能
# 部分分词器本身对某些命名实体识别效果较好
HanLP.Config.japaneseNameRecognize = False

viterbiNewSegment = HanLP.newSegment("viterbi").enableJapaneseNameRecognize(True)
CRFnewSegment_new = HanLP.newSegment("crf").enableJapaneseNameRecognize(True)
# segSentence
# CRFnewSegment_2.seg2sentence(sentences)
for sentence in sentences:
    print("crf : ",CRFnewSegment.seg(sentence))
    print("crf_new : ",CRFnewSegment_new.seg(sentence))
    print("viterbi : ",viterbiNewSegment.seg(sentence))
```

    [译智社/n, 的/u, 田丰/nr, 要/v, 说/v, 的/u, 是/v, 这/r, 只/d, 是/v, 一个/m, hanlp命名/vn, 实体/n, 识别/v, 的/u, 例子/n]
    
    ========== 命名实体开启与关闭对比试验 ==========
    
    crf :  [北川/ns, 景子/n, 参演/v, 了/u, 林诣彬/nr, 导演/n, 的/u, 《/w, 速度/n, 与/c, 激情/n, 3/m, 》/w]
    crf_new :  [北川/ns, 景子/n, 参演/v, 了/u, 林诣彬/nr, 导演/n, 的/u, 《/w, 速度/n, 与/c, 激情/n, 3/m, 》/w]
    viterbi :  [北川景子/nrj, 参演/v, 了/ule, 林诣彬/nr, 导演/nnt, 的/ude1, 《/w, 速度/n, 与/cc, 激情/n, 3/m, 》/w]
    crf :  [林志玲/nr, 亮相/v, 网友/n, :/w, 确定/v, 不/d, 是/v, 波多野/n, 结衣/n, ？/w]
    crf_new :  [林志玲/nr, 亮相/v, 网友/n, :/w, 确定/v, 不/d, 是/v, 波多野/n, 结衣/n, ？/w]
    viterbi :  [林志玲/nr, 亮相/vi, 网友/n, :/w, 确定/v, 不是/c, 波多野结衣/nrj, ？/w]
    crf :  [龟/v, 山/n, 千/m, 广/q, 和/c, 近藤/a, 公园/n, 在/p, 龟山公园/ns, 里/f, 喝/v, 酒/n, 赏/v, 花/n]
    crf_new :  [龟/v, 山/n, 千/m, 广/q, 和/c, 近藤/a, 公园/n, 在/p, 龟山公园/ns, 里/f, 喝/v, 酒/n, 赏/v, 花/n]
    viterbi :  [龟山千广/nrj, 和/cc, 近藤公园/nrj, 在/p, 龟山/nz, 公园/n, 里/f, 喝酒/vi, 赏花/nz]


## 正式内容
## 中国人名识别

说明
   - 目前分词器基本上都默认开启了中国人名识别，比如HanLP.segment()接口中使用的分词器等等，用户不必手动开启；上面的代码只是为了强调。
   - 有一定的误命中率，比如误命中关键年，则可以通过在data/dictionary/person/nr.txt加入一条关键年 A 1来排除关键年作为人名的可能性，也可以将关键年作为新词登记到自定义词典中。
   - 如果你通过上述办法解决了问题，欢迎向我提交pull request，词典也是宝贵的财富。
   - 建议NLP用户使用感知机或CRF词法分析器，精度更高。
   
算法详解
   - [《实战HMM-Viterbi角色标注中国人名识别》](http://www.hankcs.com/nlp/chinese-name-recognition-in-actual-hmm-viterbi-role-labeling.html)


```python
# 中文人名识别
def demo_chinese_name_recognition(sentences):
    segment = HanLP.newSegment().enableNameRecognize(True);
    for sentence in sentences:
        term_list = segment.seg(sentence)
        print(term_list)
        print([i.word for i in term_list])


sentences = [
    "签约仪式前，秦光荣、李纪恒、仇和等一同会见了参加签约的企业家。",
    "武大靖创世界纪录夺冠，中国代表团平昌首金",
    "区长庄木弟新年致辞",
    "朱立伦：两岸都希望共创双赢 习朱历史会晤在即",
    "陕西首富吴一坚被带走 与令计划妻子有交集",
    "据美国之音电台网站4月28日报道，8岁的凯瑟琳·克罗尔（凤甫娟）和很多华裔美国小朋友一样，小小年纪就开始学小提琴了。她的妈妈是位虎妈么？",
    "凯瑟琳和露西（庐瑞媛），跟她们的哥哥们有一些不同。",
    "王国强、高峰、汪洋、张朝阳光着头、韩寒、小四",
    "张浩和胡健康复员回家了",
    "王总和小丽结婚了",
    "编剧邵钧林和稽道青说",
    "这里有关天培的有关事迹",
    "龚学平等领导说,邓颖超生前杜绝超生",]
demo_chinese_name_recognition(sentences)

print("\n========== 中文人名 基本默认已开启 ==========\n")
print(CRFnewSegment.seg(sentences[0]))
```

    [签约/vi, 仪式/n, 前/f, ，/w, 秦光荣/nr, 、/w, 李纪恒/nr, 、/w, 仇和/nr, 等/udeng, 一同/d, 会见/v, 了/ule, 参加/v, 签约/vi, 的/ude1, 企业家/nnt, 。/w]
    ['签约', '仪式', '前', '，', '秦光荣', '、', '李纪恒', '、', '仇和', '等', '一同', '会见', '了', '参加', '签约', '的', '企业家', '。']
    [武大靖/nr, 创/v, 世界/n, 纪录/n, 夺冠/vi, ，/w, 中国/ns, 代表团/n, 平昌/ns, 首/q, 金/b]
    ['武大靖', '创', '世界', '纪录', '夺冠', '，', '中国', '代表团', '平昌', '首', '金']
    [区长/nnt, 庄木弟/nr, 新年/t, 致辞/vi]
    ['区长', '庄木弟', '新年', '致辞']
    [朱立伦/nr, ：/w, 两岸/n, 都/d, 希望/v, 共创/v, 双赢/n,  /w, 习/v, 朱/ag, 历史/n, 会晤/vn, 在即/vi]
    ['朱立伦', '：', '两岸', '都', '希望', '共创', '双赢', ' ', '习', '朱', '历史', '会晤', '在即']
    [陕西/ns, 首富/n, 吴一坚/nr, 被/pbei, 带走/v,  /w, 与/cc, 令计划/nr, 妻子/n, 有/vyou, 交集/v]
    ['陕西', '首富', '吴一坚', '被', '带走', ' ', '与', '令计划', '妻子', '有', '交集']
    [据/p, 美国之音/n, 电台/nis, 网站/n, 4月/t, 28/m, 日/b, 报道/v, ，/w, 8/m, 岁/qt, 的/ude1, 凯瑟琳/nr, ·/w, 克/q, 罗尔/nr, （/w, 凤甫娟/nr, ）/w, 和/cc, 很多/m, 华裔/n, 美国/nsf, 小朋友/n, 一样/uyy, ，/w, 小小/z, 年纪/n, 就/d, 开始/v, 学/v, 小提琴/n, 了/ule, 。/w, 她/rr, 的/ude1, 妈妈/n, 是/vshi, 位/q, 虎妈/nz, 么/y, ？/w]
    ['据', '美国之音', '电台', '网站', '4月', '28', '日', '报道', '，', '8', '岁', '的', '凯瑟琳', '·', '克', '罗尔', '（', '凤甫娟', '）', '和', '很多', '华裔', '美国', '小朋友', '一样', '，', '小小', '年纪', '就', '开始', '学', '小提琴', '了', '。', '她', '的', '妈妈', '是', '位', '虎妈', '么', '？']
    [凯瑟琳/nr, 和/cc, 露西/nr, （/w, 庐瑞媛/nr, ）/w, ，/w, 跟/p, 她们/rr, 的/ude1, 哥哥/n, 们/k, 有/vyou, 一些/m, 不同/a, 。/w]
    ['凯瑟琳', '和', '露西', '（', '庐瑞媛', '）', '，', '跟', '她们', '的', '哥哥', '们', '有', '一些', '不同', '。']
    [王国强/nr, 、/w, 高峰/n, 、/w, 汪洋/n, 、/w, 张朝阳/nr, 光/n, 着/uzhe, 头/n, 、/w, 韩寒/nr, 、/w, 小/a, 四/m]
    ['王国强', '、', '高峰', '、', '汪洋', '、', '张朝阳', '光', '着', '头', '、', '韩寒', '、', '小', '四']
    [张浩/nr, 和/cc, 胡健康/nr, 复员/v, 回家/vi, 了/ule]
    ['张浩', '和', '胡健康', '复员', '回家', '了']
    [王总/nr, 和/cc, 小丽/nr, 结婚/vi, 了/ule]
    ['王总', '和', '小丽', '结婚', '了']
    [编剧/nnt, 邵钧林/nr, 和/cc, 稽道青/nr, 说/v]
    ['编剧', '邵钧林', '和', '稽道青', '说']
    [这里/rzs, 有/vyou, 关天培/nr, 的/ude1, 有关/vn, 事迹/n]
    ['这里', '有', '关天培', '的', '有关', '事迹']
    [龚学平/nr, 等/udeng, 领导/n, 说/v, ,/w, 邓颖超/nr, 生前/t, 杜绝/v, 超生/vi]
    ['龚学平', '等', '领导', '说', ',', '邓颖超', '生前', '杜绝', '超生']
    
    ========== 中文人名 基本默认已开启 ==========
    
    [签约/vn, 仪式/n, 前/f, ，/w, 秦光荣/nr, 、/w, 李纪恒/nr, 、/w, 仇和/nr, 等/u, 一同/d, 会见/v, 了/u, 参加/v, 签约/v, 的/u, 企业家/n, 。/w]


## 音译人名识别

说明
   - 目前分词器基本上都默认开启了音译人名识别，用户不必手动开启；上面的代码只是为了强调。

算法详解
   - [《层叠隐马模型下的音译人名和日本人名识别》](http://www.hankcs.com/nlp/name-transliteration-cascaded-hidden-markov-model-and-japanese-personal-names-recognition.html)


```python
# 音译人名识别
sentences = [
    "一桶冰水当头倒下，微软的比尔盖茨、Facebook的扎克伯格跟桑德博格、亚马逊的贝索斯、苹果的库克全都不惜湿身入镜，这些硅谷的科技人，飞蛾扑火似地牺牲演出，其实全为了慈善。",
    "世界上最长的姓名是简森·乔伊·亚历山大·比基·卡利斯勒·达夫·埃利奥特·福克斯·伊维鲁莫·马尔尼·梅尔斯·帕特森·汤普森·华莱士·普雷斯顿。",
]

segment = HanLP.newSegment().enableTranslatedNameRecognize(True)
for sentence in sentences:
    term_list = segment.seg(sentence)
    print(term_list)
    
print("\n========== 音译人名 默认已开启 ==========\n")
print(CRFnewSegment.seg(sentences[0]))
```

    [一桶/nz, 冰水/n, 当头/vi, 倒下/v, ，/w, 微软/ntc, 的/ude1, 比尔盖茨/nrf, 、/w, Facebook/nx, 的/ude1, 扎克伯格/nr, 跟/p, 桑德博格/nrf, 、/w, 亚马逊/nrf, 的/ude1, 贝索斯/nrf, 、/w, 苹果/nf, 的/ude1, 库克/nr, 全都/d, 不惜/v, 湿身/nz, 入镜/nz, ，/w, 这些/rz, 硅谷/ns, 的/ude1, 科技/n, 人/n, ，/w, 飞蛾/n, 扑火/vn, 似/vg, 地/ude2, 牺牲/v, 演出/vn, ，/w, 其实/d, 全/a, 为了/p, 慈善/a, 。/w]
    [世界/n, 上/f, 最长/d, 的/ude1, 姓名/n, 是/vshi, 简森/nr, ·/w, 乔伊/nr, ·/w, 亚历山大/nr, ·/w, 比基/nr, ·/w, 卡利斯/nr, 勒/v, ·/w, 达夫·埃利奥特·福克斯·伊维鲁莫·马尔尼·梅尔斯·帕特森·汤普森·华莱士·普雷斯顿/nrf, 。/w]
    
    ========== 音译人名 默认已开启 ==========
    
    [一桶/m, 冰水/n, 当头/d, 倒下/v, ，/w, 微软/a, 的/u, 比尔盖茨/n, 、/w, Facebook/l, 的/u, 扎克伯格/n, 跟/p, 桑德博格/n, 、/w, 亚马逊/nr, 的/u, 贝索斯/nr, 、/w, 苹果/n, 的/u, 库克/nr, 全都/d, 不惜/v, 湿身/n, 入镜/v, ，/w, 这些/r, 硅谷/n, 的/u, 科技/n, 人/n, ，/w, 飞蛾/v, 扑火似/v, 地/u, 牺牲/v, 演出/v, ，/w, 其实/d, 全/d, 为了/p, 慈善/a, 。/w]


## 日本人名识别

说明
   - 目前标准分词器默认关闭了日本人名识别，用户需要手动开启；这是因为日本人名的出现频率较低，但是又消耗性能。

算法详解
   - [《层叠隐马模型下的音译人名和日本人名识别》](http://www.hankcs.com/nlp/name-transliteration-cascaded-hidden-markov-model-and-japanese-personal-names-recognition.html)


```python
# 日语人名识别
def demo_japanese_name_recognition(sentences):

    segment = HanLP.newSegment().enableJapaneseNameRecognize(True)
    for sentence in sentences:
        term_list = segment.seg(sentence)
        print(term_list)
        print([i.word for i in term_list])
        

sentences =[
    "北川景子参演了林诣彬导演的《速度与激情3》",
    "林志玲亮相网友:确定不是波多野结衣？",
    "龟山千广和近藤公园在龟山公园里喝酒赏花",
 ]
demo_japanese_name_recognition(sentences)
print("\n========== 日文人名 标准分词器默认未开启 ==========\n")
print(CRFnewSegment.seg(sentences[0]))
```

    [北川景子/nrj, 参演/v, 了/ule, 林诣彬/nr, 导演/nnt, 的/ude1, 《/w, 速度/n, 与/cc, 激情/n, 3/m, 》/w]
    ['北川景子', '参演', '了', '林诣彬', '导演', '的', '《', '速度', '与', '激情', '3', '》']
    [林志玲/nr, 亮相/vi, 网友/n, :/w, 确定/v, 不是/c, 波多野结衣/nrj, ？/w]
    ['林志玲', '亮相', '网友', ':', '确定', '不是', '波多野结衣', '？']
    [龟山千广/nrj, 和/cc, 近藤公园/nrj, 在/p, 龟山/nz, 公园/n, 里/f, 喝酒/vi, 赏花/nz]
    ['龟山千广', '和', '近藤公园', '在', '龟山', '公园', '里', '喝酒', '赏花']
    
    ========== 日文人名 标准分词器默认未开启 ==========
    
    [北川/ns, 景子/n, 参演/v, 了/u, 林诣彬/nr, 导演/n, 的/u, 《/w, 速度/n, 与/c, 激情/n, 3/m, 》/w]


## 地名识别
说明
   - 目前标准分词器都默认关闭了地名识别，用户需要手动开启；这是因为消耗性能，其实多数地名都收录在核心词典和用户自定义词典中。
   - 在生产环境中，能靠词典解决的问题就靠词典解决，这是最高效稳定的方法。
   - 建议对命名实体识别要求较高的用户使用感知机词法分析器。
   
   
算法详解
   - [《实战HMM-Viterbi角色标注地名识别》](http://www.hankcs.com/nlp/ner/place-names-to-identify-actual-hmm-viterbi-role-labeling.html)


```python
# 演示数词与数量词识别
sentences = [
    "十九元套餐包括什么",
    "九千九百九十九朵玫瑰",    
    "壹佰块都不给我",
    "９０１２３４５６７８只蚂蚁",
    "牛奶三〇〇克*2",
    "ChinaJoy“扫黄”细则露胸超2厘米罚款",
]

StandardTokenizer = JClass("com.hankcs.hanlp.tokenizer.StandardTokenizer")

StandardTokenizer.SEGMENT.enableNumberQuantifierRecognize(True)
for sentence in sentences:
    print(StandardTokenizer.segment(sentence))
    
print("\n========== 演示数词与数量词 默认未开启 ==========\n")
CRFnewSegment.enableNumberQuantifierRecognize(True)
print(CRFnewSegment.seg(sentences[0]))
```

    [十九元/mq, 套餐/n, 包括/v, 什么/ry]
    [九千九百九十九朵/mq, 玫瑰/n]
    [壹佰块/mq, 都/d, 不/d, 给/p, 我/rr]
    [９０１２３４５６７８只/mq, 蚂蚁/n]
    [牛奶/nf, 三〇〇克/mq, */w, 2/m]
    [ChinaJoy/nx, “/w, 扫黄/vi, ”/w, 细则/n, 露/v, 胸/ng, 超/v, 2厘米/mq, 罚款/vi]
    
    ========== 演示数词与数量词 默认未开启 ==========
    
    [十九/m, 元/q, 套餐/n, 包括/v, 什么/r]


## 机构名识别
**说明**
 - 目前分词器默认关闭了机构名识别，用户需要手动开启；这是因为消耗性能，其实常用机构名都收录在核心词典和用户自定义词典中。
 - HanLP的目的不是演示动态识别，在生产环境中，能靠词典解决的问题就靠词典解决，这是最高效稳定的方法。
 - 建议对命名实体识别要求较高的用户使用感知机词法分析器。
 
 
**算法详解**

 - [《层叠HMM-Viterbi角色标注模型下的机构名识别》](http://www.hankcs.com/nlp/ner/place-name-recognition-model-of-the-stacked-hmm-viterbi-role-labeling.html)


```python
# 机构名识别
sentences = [
    "我在上海林原科技有限公司兼职工作，",
    "我经常在台川喜宴餐厅吃饭，",
    "偶尔去开元地中海影城看电影。",
]

Segment = JClass("com.hankcs.hanlp.seg.Segment")
Term = JClass("com.hankcs.hanlp.seg.common.Term")

segment = HanLP.newSegment().enableOrganizationRecognize(True)
for sentence in sentences:
    term_list = segment.seg(sentence)
    print(term_list)
    
print("\n========== 机构名 标准分词器已经全部关闭 ==========\n")
print(CRFnewSegment.seg(sentences[0]))

segment = HanLP.newSegment('crf').enableOrganizationRecognize(True)
```

    [我/rr, 在/p, 上海/ns, 林原科技有限公司/nt, 兼职/vn, 工作/vn, ，/w]
    [我/rr, 经常/d, 在/p, 台川喜宴餐厅/nt, 吃饭/vi, ，/w]
    [偶尔/d, 去/vf, 开元地中海影城/nt, 看/v, 电影/n, 。/w]
    
    ========== 机构名 标准分词器已经全部关闭 ==========
    
    [我/r, 在/p, 上海林原科技有限公司/nt, 兼职/vn, 工作/vn, ，/w]


## 地名识别

说明
   - 目前标准分词器都默认关闭了地名识别，用户需要手动开启；这是因为消耗性能，其实多数地名都收录在核心词典和用户自定义词典中。
   - 在生产环境中，能靠词典解决的问题就靠词典解决，这是最高效稳定的方法。
   - 建议对命名实体识别要求较高的用户使用感知机词法分析器。
    
算法详解
   - [《实战HMM-Viterbi角色标注地名识别》](http://www.hankcs.com/nlp/ner/place-names-to-identify-actual-hmm-viterbi-role-labeling.html)


```python
# 地名识别
def demo_place_recognition(sentences):
    
    segment = HanLP.newSegment().enablePlaceRecognize(True)
    for sentence in sentences:
        term_list = segment.seg(sentence)
        print(term_list)
        print([i.word for i in term_list])
        
sentences = ["蓝翔给宁夏固原市彭阳县红河镇黑牛沟村捐赠了挖掘机"]
demo_place_recognition(sentences)

print("\n========== 地名 默认已开启 ==========\n")
print(CRFnewSegment.seg(sentences[0]))
```

    [蓝翔/nr, 给/p, 宁夏/ns, 固原市/ns, 彭阳县/ns, 红河镇/ns, 黑牛沟村/ns, 捐赠/v, 了/ule, 挖掘机/n]
    ['蓝翔', '给', '宁夏', '固原市', '彭阳县', '红河镇', '黑牛沟村', '捐赠', '了', '挖掘机']
    
    ========== 地名 默认已开启 ==========
    
    [蓝翔/v, 给/v, 宁夏/ns, 固原市/ns, 彭阳县/ns, 红河镇/ns, 黑牛沟村/ns, 捐赠/v, 了/u, 挖掘机/n]


## URL 识别

自动识别URL,该部分是在demo中发现的，但是原作者并没有在文档中提到这个，该部分可以发现URL，测试发现其他分类器应该是默认不开启这个的，而且config中并没有开启该功能的选项，因此这应该是一个额外的类。我建议如果有需要的，你可以尝试先利用URLTokenizer获取URL，然后添加进用户词典。或者直接使用其他工具或者自定义函数解决该问题。


```python
# URL 识别
text = '''HanLP的项目地址是https://github.com/hankcs/HanLP，
        发布地址是https://github.com/hankcs/HanLP/releases，
        我有时候会在www.hankcs.com上面发布一些消息，
        我的微博是http://weibo.com/hankcs/，会同步推送hankcs.com的新闻。
        听说.中国域名开放申请了,但我并没有申请hankcs.中国,因为穷……
             '''

Nature = SafeJClass("com.hankcs.hanlp.corpus.tag.Nature")
Term = SafeJClass("com.hankcs.hanlp.seg.common.Term")
URLTokenizer = SafeJClass("com.hankcs.hanlp.tokenizer.URLTokenizer")

term_list = URLTokenizer.segment(text)
print(term_list)
for term in term_list:
    if term.nature == Nature.xu:
        print(term.word)
```

    [HanLP/nx, 的/ude1, 项目/n, 地址/n, 是/vshi, https://github.com/hankcs/HanLP/xu, ，/w, 
    /w,         /w, 发布/v, 地址/n, 是/vshi, https://github.com/hankcs/HanLP/releases/xu, ，/w, 
    /w,         /w, 我/rr, 有时候/d, 会/v, 在/p, www.hankcs.com/xu, 上面/f, 发布/v, 一些/m, 消息/n, ，/w, 
    /w,         /w, 我/rr, 的/ude1, 微博/n, 是/vshi, http://weibo.com/hankcs//xu, ，/w, 会/v, 同步/vd, 推送/nz, hankcs.com/xu, 的/ude1, 新闻/n, 。/w, 
    /w,         /w, 听说/v, ./w, 中国/ns, 域名/n, 开放/v, 申请/v, 了/ule, ,/w, 但/c, 我/rr, 并/cc, 没有/v, 申请/v, hankcs.中国/xu, ,/w, 因为/c, 穷/a, ……/w, 
    /w,              /w]
    https://github.com/hankcs/HanLP
    https://github.com/hankcs/HanLP/releases
    www.hankcs.com
    http://weibo.com/hankcs/
    hankcs.com
    hankcs.中国

