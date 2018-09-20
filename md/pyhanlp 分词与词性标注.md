
## pyhanlp中的分词器简介

pyhanlp实现的分词器有很多，同时pyhanlp获取hanlp中分词器也有两种方式

第一种是直接从封装好的hanlp类中获取，这种获取方式一共可以获取五种分词器，而现在默认的就是第一种维特比分词器


 * 维特比 (viterbi)：效率和效果的最佳平衡，也是最短路分词，HanLP最短路求解采用Viterbi算法
 * 双数组trie树 (dat)：极速词典分词，千万字符每秒（可能无法获取词性，此处取决于你的词典）
 * 条件随机场 (crf)：分词、词性标注与命名实体识别精度都较高，适合要求较高的NLP任务
 * 感知机 (perceptron)：分词、词性标注与命名实体识别，支持在线学习
 * N最短路 (nshort)：命名实体识别稍微好一些，牺牲了速度
 
 
第二种方式是使用JClass直接获取java类，然后使用。这种方式除了获取上面的五种分词器以外还可以获得一些其他分词器，如NLP分词器，索引分词，快速词典分词等等

## 两种使用方式的对比

第一种是使用作者给的HanLP直接获取分词器，直接segment() 会获取 默认的标准分词器也就是维特比分词器，也**可以使用newSegment函数，传入上面的分词器英文名称来获取新的分词器，如使用`HanLP.newSegment("crf")`来获取CRF分词器。**第二种方式是使用JClass从java中获取我们想要的类，好在这两种方式都比较方便。除此之外要注意的是，在pyhanlp中还给出了SafeJClass类，其为JClass的线程安全版，你也可以使用SafeClass来代替JClass。不过好在HanLP中的很多类本身已经实现了线程安全，因此许多时候两者是可以相互替代的。



```python
from pyhanlp import *
# 第一个demo

print(HanLP.segment("你好，欢迎使用HanLP汉语处理包！接下来请从其他Demo中体验HanLP丰富的功能~"))

# HanLP词性标注集: http://www.hankcs.com/nlp/part-of-speech-tagging.html#h2-8
```

    [你好/vl, ，/w, 欢迎/v, 使用/v, HanLP/nx, 汉语/gi, 处理/vn, 包/v, ！/w, 接下来/vl, 请/v, 从/p, 其他/rzv, Demo/nx, 中/f, 体验/v, HanLP/nx, 丰富/a, 的/ude1, 功能/n, ~/nx]



```python
# 标准分词
text = (
    "举办纪念活动铭记二战历史，不忘战争带给人类的深重灾难，是为了防止悲剧重演，确保和平永驻；记二战历史，更是为了提醒国际社会，需要共同捍卫二战胜利成果和国际公平正义，必须警惕和抵制在历史认知和维护战后国际秩序问题上的倒行逆施。"
)

BasicTokenizer = JClass("com.hankcs.hanlp.tokenizer.BasicTokenizer")
print(BasicTokenizer.segment(text))

import time
start = time.time()
for i in range(100000):
    HanLP.segment(text)
cost_time = time.time() - start
print("HanLP.segment ：%.2f字每秒" % (len(text) * 100000 / cost_time))

start = time.time()
for i in range(100000):
    BasicTokenizer.segment(text)
cost_time = time.time() - start
print("BasicTokenizer.segment ：%.2f字每秒" % (len(text) * 100000 / cost_time))
```

    [举办/v, 纪念活动/nz, 铭记/v, 二战/n, 历史/n, ，/w, 不忘/v, 战争/n, 带给/v, 人类/n, 的/ude1, 深重/a, 灾难/n, ，/w, 是/vshi, 为了/p, 防止/v, 悲剧/n, 重演/v, ，/w, 确保/v, 和平/n, 永驻/nz, ；/w, 记/v, 二战/n, 历史/n, ，/w, 更是/d, 为了/p, 提醒/v, 国际/n, 社会/n, ，/w, 需要/v, 共同/d, 捍卫/v, 二战/n, 胜利/vn, 成果/n, 和/cc, 国际/n, 公平/a, 正义/n, ，/w, 必须/d, 警惕/v, 和/cc, 抵制/v, 在/p, 历史/n, 认知/vn, 和/cc, 维护/v, 战后/t, 国际/n, 秩序/n, 问题/n, 上/f, 的/ude1, 倒行逆施/vl, 。/w]
    HanLP.segment ：1518389.32字每秒
    BasicTokenizer.segment ：2415039.64字每秒


仅仅从刚刚的结果看，可能会不太理解为同一个分词器性能差距这么大？难道是因为中间代码的调度问题，其实也不是。将两段代码前后互换之后，发现无论两者怎么排列，总是在前的速度较慢，在后的较快，因此应该是内存的问题，第二次调用时减少了部分内存的调动。所以同一个分词器才会出现，第二次总比第一次快的现象。

## 标准分词

- 说明
  * HanLP中有一系列“开箱即用”的静态分词器，以`Tokenizer`结尾，在接下来的例子中会继续介绍。
  * `HanLP.segment`其实是对`StandardTokenizer.segment`的包装。
  * 分词结果包含词性，每个词性的意思请查阅[《HanLP词性标注集》](http://www.hankcs.com/nlp/part-of-speech-tagging.html#h2-8)。
- 算法详解
  * [《词图的生成》](http://www.hankcs.com/nlp/segment/the-word-graph-is-generated.html)


## 单独获取词性或者词语

如你所见的是，前面print的结果是[词语/词性，词语/词性，/词语/词性......]的形式，那么如果我们只想获取词语，或者词性应该怎么办呢？

方法也很简单。使用`HanLP.Config.ShowTermNature = False`修改配置，使其不显示词性即可。

如果想要只获取词性也是可以的，因为原分词器返回的是Java中的ArrayList属性，list中的每个单元都是一个term类，因此我们也可以通过获取term中的word字段来直接获取词语，或者nature属性，直接获取词性。这一特征，我们在之后也会用到。


```python
HanLP.Config.ShowTermNature = False
text = "小区居民有的反对喂养流浪猫"
CRFnewSegment = HanLP.newSegment("crf")
term_list = CRFnewSegment.seg(text)
print(term_list)
HanLP.Config.ShowTermNature = True


CRFnewSegment = HanLP.newSegment("crf")
term_list = CRFnewSegment.seg(text)
print(term_list)
print([str(i.word) for i in term_list])
print([str(i.nature) for i in term_list])
```

    [小区, 居民, 有的, 反对, 喂, 养, 流, 浪, 猫]
    [小区/n, 居民/n, 有的/r, 反对/v, 喂/v, 养/v, 流/v, 浪/n, 猫/v]
    ['小区', '居民', '有的', '反对', '喂', '养', '流', '浪', '猫']
    ['n', 'n', 'r', 'v', 'v', 'v', 'v', 'n', 'v']


## 其他分词器的代码实现与速度对比

到这里，分词与词性标注的主要功能其实已经讲的差不多了，后边是其他分词器的代码展示以及部分其他内容，本人建议最起码通读一遍，如果只想获取词性或者词语的话可以按照刚刚的写法进行修改，如果需要了解原理则可以点开相应链接。

## NLP 分词

- 说明
  * NLP分词`NLPTokenizer`会执行词性标注和命名实体识别，由[结构化感知机序列标注框架](https://github.com/hankcs/HanLP/wiki/%E7%BB%93%E6%9E%84%E5%8C%96%E6%84%9F%E7%9F%A5%E6%9C%BA%E6%A0%87%E6%B3%A8%E6%A1%86%E6%9E%B6)支撑。
  * 默认模型训练自`9970`万字的大型综合语料库，是已知范围内**全世界最大**的中文分词语料库。语料库规模决定实际效果，面向生产环境的语料库应当在千万字量级。欢迎用户在自己的语料上[训练新模型](https://github.com/hankcs/HanLP/wiki/%E7%BB%93%E6%9E%84%E5%8C%96%E6%84%9F%E7%9F%A5%E6%9C%BA%E6%A0%87%E6%B3%A8%E6%A1%86%E6%9E%B6)以适应新领域、识别新的命名实体。


```python
# NLP分词
# NLP分词，更精准的中文分词、词性标注与命名实体识别

NLPTokenizer = JClass("com.hankcs.hanlp.tokenizer.NLPTokenizer")
print(NLPTokenizer.segment("我新造一个词叫幻想乡你能识别并正确标注词性吗？"))  # “正确”是副形词。
# 注意观察下面两个“希望”的词性、两个“晚霞”的词性
print(NLPTokenizer.analyze("我的希望是希望张晚霞的背影被晚霞映红").translateLabels())
print(NLPTokenizer.analyze("支援臺灣正體香港繁體：微软公司於1975年由比爾·蓋茲和保羅·艾倫創立。"))

print("\n==========||==========\n")

print("为了验证NLP分词与标准分词的差异,我们在用text进行一次分词")

pressure = 1000
start = time.time()
for i in range(pressure):
    BasicTokenizer.segment(text)
cost_time = time.time() - start
print("HanLP.segment ：%.2f字每秒" % (len(text) * pressure / cost_time))
```

    [我/r, 新/d, 造/v, 一个/m, 词/n, 叫/v, 幻想乡/ns, 你/r, 能/v, 识别/v, 并/c, 正确/ad, 标注/v, 词性/n, 吗/y, ？/w]
    我/代词 的/助词 希望/名词 是/动词 希望/动词 张晚霞/人名 的/助词 背影/名词 被/介词 晚霞/名词 映红/人名
    支援/v 臺灣/ns 正體/n 香港/ns 繁體/n ：/v [微软/nt 公司/n]/nt 於/p 1975年/t 由/p 比爾·蓋茲/n 和/c 保羅·艾倫/nr 創立/v 。/w
    
    ==========||==========
    
    为了验证NLP分词与标准分词的差异,我们在用text进行一次分词
    HanLP.segment ：267563.45字每秒


## 索引分词

- 说明
  * 索引分词`IndexTokenizer`是面向搜索引擎的分词器，能够对长词全切分，另外通过`term.offset`可以获取单词在文本中的偏移量。
  * 任何分词器都可以通过基类`Segment`的`enableIndexMode`方法激活索引模式。



```python
from jpype import *

# 索引分词

Term =JClass("com.hankcs.hanlp.seg.common.Term")
IndexTokenizer = JClass("com.hankcs.hanlp.tokenizer.IndexTokenizer")

term_list = IndexTokenizer.segment("主副食品")
for term in term_list.iterator():
    print("{} [{}:{}]".format(term, term.offset, term.offset + len(term.word)))

print("最细颗粒度切分：") # 此处对应jieba的搜索引擎模式
IndexTokenizer.SEGMENT.enableIndexMode(JInt(1))  # JInt用于区分重载
term_list = IndexTokenizer.segment("主副食品")
for term in term_list.iterator():
    print("{} [{}:{}]".format(term, term.offset, term.offset + len(term.word)))

print("\n==========||==========\n")

print("现在索引分词对text进行一次分词")

import time
start = time.time()
pressure = 100000
for i in range(pressure):
    IndexTokenizer.segment(text)
cost_time = time.time() - start
print("IndexTokenizer.SEGMENT.enableIndexMode(JInt(1))-> 分词速度：%.2f字每秒" % (len(text) * pressure / cost_time))
```

    主副食品/n [0:4]
    主副食/j [0:3]
    主/ag [0:1]
    副食品/n [1:4]
    副食/n [1:3]
    副/b [1:2]
    食品/n [2:4]
    食/v [2:3]
    品/ng [3:4]
    最细颗粒度切分：
    主副食品/n [0:4]
    主副食/j [0:3]
    主/ag [0:1]
    副食品/n [1:4]
    副食/n [1:3]
    副/b [1:2]
    食品/n [2:4]
    食/v [2:3]
    品/ng [3:4]
    
    ==========||==========
    
    现在索引分词对text进行一次分词
    IndexTokenizer.SEGMENT.enableIndexMode(JInt(1))-> 分词速度：494700.58字每秒


##  最短路分词&N最短路分词（写法一）

- 说明
  * N最短路分词器`NShortSegment`比最短路分词器慢，但是效果稍微好一些，对命名实体识别能力更强。
  * 一般场景下最短路分词的精度已经足够，而且速度比N最短路分词器快几倍，请酌情选择。
- 算法详解
  * [《N最短路径的Java实现与分词应用》](http://www.hankcs.com/nlp/segment/n-shortest-path-to-the-java-implementation-and-application-segmentation.html)


```python
sentences = [
    "今天，刘志军案的关键人物,山西女商人丁书苗在市二中院出庭受审。",
    "江西省监狱管理局与中国太平洋财产保险股份有限公司南昌中心支公司保险合同纠纷案",
    "新北商贸有限公司",
]

NShortNewSegment = HanLP.newSegment("nshort")

for sentence in sentences:
    term_list = NShortNewSegment.seg(sentence)
    print(term_list)
    
import time
pressure = 10000

start = time.time()
for i in range(pressure):
    NShortNewSegment.seg(text)
cost_time = time.time() - start
print("NShortNewSegment 分词速度：%.2f字每秒" % (len(text) * pressure / cost_time))
```

    [今天/t, ，/w, 刘志军/nr, 案/ng, 的/ude1, 关键/n, 人物/n, ,/w, 山西/ns, 女/b, 商人/nnt, 丁书苗/nr, 在/p, 市/n, 二/m, 中/f, 院/n, 出庭/vi, 受审/vi, 。/w]
    [江西省/ns, 监狱/nis, 管理局/nis, 与/cc, 中国/ns, 太平洋/ns, 财产/n, 保险/n, 股份/n, 有限公司/nis, 南昌/ns, 中心/nis, 支公司/nis, 保险/n, 合同/n, 纠纷案/nz]
    [新/a, 北/f, 商贸/n, 有限公司/nis]
    NShortNewSegment 分词速度：208784.50字每秒


##  最短路分词&N最短路分词（写法二）


```python
# 最短路分词&N最短路分词

NShortSegment = JClass("com.hankcs.hanlp.seg.NShort.NShortSegment")
Segment = JClass("com.hankcs.hanlp.seg.Segment")
ViterbiSegment = JClass("com.hankcs.hanlp.seg.Viterbi.ViterbiSegment")

nshort_segment = NShortSegment().enableCustomDictionary(
    False).enablePlaceRecognize(True).enableOrganizationRecognize(True)
shortest_segment = ViterbiSegment().enableCustomDictionary(
    False).enablePlaceRecognize(True).enableOrganizationRecognize(True)

for sentence in sentences:
    print("N-最短分词：{} \n最短路分词：{}".format(
        nshort_segment.seg(sentence), shortest_segment.seg(sentence)))
    
    
print("\n==========||==========\n")

print("下面对最短路径分词器与N最短路径分词器进行速度对比")
import time
pressure = 10000

start = time.time()
for i in range(pressure):
    nshort_segment.seg(text)
cost_time = time.time() - start
print("nshort_segment分词速度：%.2f字每秒" % (len(text) * pressure / cost_time))

start = time.time()
for i in range(pressure):
    shortest_segment.seg(text)
cost_time = time.time() - start
print("shortest_segment分词速度：%.2f字每秒" % (len(text) * pressure / cost_time))
```

    N-最短分词：[今天/t, ，/w, 刘志军/nr, 案/ng, 的/ude1, 关键/n, 人物/n, ,/w, 山西/ns, 女/b, 商人/nnt, 丁书苗/nr, 在/p, 市二中院/nt, 出庭/vi, 受审/vi, 。/w] 
    最短路分词：[今天/t, ，/w, 刘志军/nr, 案/ng, 的/ude1, 关键/n, 人物/n, ,/w, 山西/ns, 女/b, 商人/nnt, 丁书苗/nr, 在/p, 市/n, 二中院/nt, 出庭/vi, 受审/vi, 。/w]
    N-最短分词：[江西省监狱管理局/nt, 与/cc, 中国/ns, 太平洋/ns, 财产保险股份有限公司南昌中心支公司/nt, 保险/n, 合同/n, 纠纷案/nz] 
    最短路分词：[江西省监狱管理局/nt, 与/cc, 中国/ns, 太平洋/ns, 财产保险股份有限公司南昌中心支公司/nt, 保险/n, 合同/n, 纠纷案/nz]
    N-最短分词：[新北商贸有限公司/nt] 
    最短路分词：[新北商贸有限公司/nt]
    
    ==========||==========
    
    下面对最短路径分词器与N最短路径分词器进行速度对比
    nshort_segment分词速度：140187.31字每秒
    shortest_segment分词速度：321353.36字每秒


## CRF分词

- 说明
  * CRF对新词有很好的识别能力，但是开销较大。
- 算法详解
  * [《CRF中文分词、词性标注与命名实体识别》](https://github.com/hankcs/HanLP/wiki/CRF%E8%AF%8D%E6%B3%95%E5%88%86%E6%9E%90)



```python
# CRF分词
# 原来的CRF分词已经无法使用,架构发生了较大的改变
# 我们这里使用新的架构,新的语法调用"link com.hankcs.hanlp.model.crf.CRFLexicalAnalyzer"来进行分词

# 原作者为了能够在将来HanLP升级时,能够一直获取合适的分类器
# 创建了 segment 的工厂方法,以下我写的示范语法是通用的
# 关于更多newSegment 的参数,可以参考源码中的HanLP.java
# segment  则可以参考 Segment类
# 下面是原作者写在 HanLP.java 里的原文
"""
    /**
     * 创建一个分词器，
     * 这是一个工厂方法<br>
     *
     * @param algorithm 分词算法，传入算法的中英文名都可以，可选列表：<br>
     *                  <ul>
     *                  <li>维特比 (viterbi)：效率和效果的最佳平衡</li>
     *                  <li>双数组trie树 (dat)：极速词典分词，千万字符每秒</li>
     *                  <li>条件随机场 (crf)：分词、词性标注与命名实体识别精度都较高，适合要求较高的NLP任务</li>
     *                  <li>感知机 (perceptron)：分词、词性标注与命名实体识别，支持在线学习</li>
     *                  <li>N最短路 (nshort)：命名实体识别稍微好一些，牺牲了速度</li>
     *                  <li>2阶隐马 (hmm2)：训练速度较CRF快</li> (已经废除,笔者FT补充)
     *                  </ul>
     * @return 一个分词器
     */
"""

sentence_array = [
    "HanLP是由一系列模型与算法组成的Java工具包，目标是普及自然语言处理在生产环境中的应用。",
    "鐵桿部隊憤怒情緒集結 馬英九腹背受敵",  # 繁体无压力
    "馬英九回應連勝文“丐幫說”：稱黨內同志談話應謹慎",
    "高锰酸钾，强氧化剂，紫红色晶体，可溶于水，遇乙醇即被还原。常用作消毒剂、水净化剂、氧化剂、漂白剂、毒气吸收剂、二氧化碳精制剂等。",
    "《夜晚的骰子》通过描述浅草的舞女在暗夜中扔骰子的情景,寄托了作者对庶民生活区的情感",
    "这个像是真的[委屈]前面那个打扮太江户了，一点不上品...@hankcs",
    "鼎泰丰的小笼一点味道也没有...每样都淡淡的...淡淡的，哪有食堂2A的好次",
    "克里斯蒂娜·克罗尔说：不，我不是虎妈。我全家都热爱音乐，我也鼓励他们这么做。",
    "今日APPS：Sago Mini Toolbox培养孩子动手能力",
    "财政部副部长王保安调任国家统计局党组书记",
    "2.34米男子娶1.53米女粉丝 称夫妻生活没问题",
    "你看过穆赫兰道吗",
    "国办发布网络提速降费十四条指导意见 鼓励流量不清零",
    "乐视超级手机能否承载贾布斯的生态梦"
]


CRFnewSegment = HanLP.newSegment("crf")

for sentence in sentence_array:
    term_list = CRFnewSegment.seg(sentence)
    print(term_list)
    
import time
pressure = 10000

start = time.time()
for i in range(pressure):
    CRFnewSegment.seg(text)
cost_time = time.time() - start
print("CRFnewSegment分词速度：%.2f字每秒" % (len(text) * pressure / cost_time))
```

    [HanLP/nx, 是/v, 由/p, 一/m, 系列/q, 模型/n, 与/c, 算法/n, 组成/v, 的/u, Java/vn, 工具/n, 包/v, ，/w, 目标/n, 是/v, 普及/v, 自然语言处理/v, 在/p, 生产/vn, 环境/n, 中的/vn, 应用/vn, 。/w]
    [鐵桿/n, 部隊/n, 憤怒/vn, 情緒/n, 集結/vn,  馬英九/vn, 腹/Ng, 背/v, 受/v, 敵/Ng]
    [馬英九/nr, 回/v, 應/v, 連/u, 勝/v, 文/Ng, “/w, 丐幫/nr, 說/v, ”/w, ：/w, 稱/v, 黨內/s, 同志/n, 談話/v, 應/v, 謹慎/a]
    [高锰/nr, 酸钾/n, ，/w, 强氧化剂/n, ，/w, 紫红色/n, 晶体/n, ，/w, 可/v, 溶于/v, 水/n, ，/w, 遇乙醇/v, 即/v, 被/p, 还原/v, 。/w, 常/d, 用/p, 作/v, 消毒剂/n, 、/w, 水/n, 净化剂/n, 、/w, 氧化剂/n, 、/w, 漂白剂/n, 、/w, 毒气/n, 吸收剂/n, 、/w, 二氧化碳/n, 精制剂/n, 等/u, 。/w]
    [《/w, 夜晚/t, 的/u, 骰子/n, 》/w, 通过/p, 描述/v, 浅草/n, 的/u, 舞女/n, 在/p, 暗夜/n, 中/f, 扔/v, 骰子/n, 的/u, 情景/n, ,/w, 寄托/v, 了/u, 作者/n, 对/p, 庶民/n, 生活区/n, 的/u, 情感/n]
    [这个/r, 像/v, 是/v, 真的/d, [/w, 委屈/n, ]/w, 前面/f, 那个/r, 打扮/v, 太江户/n, 了/y, ，/w, 一点/m, 不/d, 上/v, 品...@hankcs/n]
    [鼎泰丰/nr, 的/u, 小笼/n, 一点/m, 味道/n, 也/d, 没有/d, .../v, 每样/r, 都/d, 淡淡/a, 的/u, .../n, 淡淡/a, 的/u, ，/w, 哪/r, 有/v, 食堂/n, 2A/m, 的/u, 好/a, 次/Bg]
    [克里斯蒂娜·克罗尔/nr, 说/v, ：/w, 不/d, ，/w, 我/r, 不/d, 是/v, 虎妈/n, 。/w, 我/r, 全家/n, 都/d, 热爱/v, 音乐/n, ，/w, 我/r, 也/d, 鼓励/v, 他们/r, 这么/r, 做/v, 。/w]
    [今日/t, A/nx, PPS/n, ：/w, Sago Mini Toolbox/nx, 培养/v, 孩子/n, 动手/v, 能力/n]
    [财政部/nt, 副/b, 部长/n, 王保安/nr, 调任/v, 国家统计局党组/nt, 书记/n]
    [2.34/m, 米/q, 男子/n, 娶/v, 1.53/m, 米/q, 女/b, 粉丝 /n, 称/v, 夫妻/n, 生活/vn, 没/v, 问题/n]
    [你/r, 看/v, 过/u, 穆赫兰道/n, 吗/y]
    [国办/j, 发布/v, 网络/n, 提/v, 速/Ng, 降费/n, 十四/m, 条/q, 指导/vn, 意见/n,  /v, 鼓励/v, 流量/n, 不/d, 清/v, 零/m]
    [乐视/v, 超级/b, 手机/n, 能否/v, 承载/v, 贾布斯/nr, 的/u, 生态梦/n]
    CRFnewSegment分词速度：38895.99字每秒


## 极速词典分词

- 说明
  * 极速分词是词典最长分词，速度极其快，精度一般。
  * 在i7-6700K上跑出了4500万字每秒的速度。
- 算法详解
  * [《Aho Corasick自动机结合DoubleArrayTrie极速多模式匹配》](http://www.hankcs.com/program/algorithm/aho-corasick-double-array-trie.html)



```python
DATnewSegment = HanLP.newSegment("dat")

for sentence in sentence_array:
    term_list = DATnewSegment.seg(sentence)
    print(term_list)
    
import time
pressure = 10000

start = time.time()
for i in range(pressure):
    DATnewSegment.seg(text)
cost_time = time.time() - start
print("DATnewSegment 分词速度：%.2f字每秒" % (len(text) * pressure / cost_time))
```

    [H/null, a/null, n/null, L/null, P/null, 是/null, 由/null, 一系列/null, 模型/null, 与/null, 算法/null, 组成/null, 的/null, J/null, a/null, v/null, a/null, 工具包/null, ，/null, 目标/null, 是/null, 普及/null, 自然/null, 语言/null, 处理/null, 在/null, 生产/null, 环境/null, 中/null, 的/null, 应用/null, 。/null]
    [鐵/null, 桿/null, 部/null, 隊/null, 憤/null, 怒/null, 情/null, 緒/null, 集/null, 結/null,  /null, 馬/null, 英/null, 九/null, 腹/null, 背/null, 受/null, 敵/null]
    [馬/null, 英/null, 九/null, 回/null, 應/null, 連/null, 勝/null, 文/null, “/null, 丐/null, 幫/null, 說/null, ”/null, ：/null, 稱/null, 黨/null, 內/null, 同志/null, 談/null, 話/null, 應/null, 謹/null, 慎/null]
    [高锰酸钾/null, ，/null, 强氧化剂/null, ，/null, 紫红色/null, 晶体/null, ，/null, 可溶/null, 于/null, 水/null, ，/null, 遇/null, 乙醇/null, 即/null, 被/null, 还原/null, 。/null, 常用/null, 作/null, 消毒剂/null, 、/null, 水/null, 净化/null, 剂/null, 、/null, 氧化剂/null, 、/null, 漂白剂/null, 、/null, 毒气/null, 吸收剂/null, 、/null, 二氧化碳/null, 精制/null, 剂/null, 等/null, 。/null]
    [《/null, 夜晚/null, 的/null, 骰子/null, 》/null, 通过/null, 描述/null, 浅草/null, 的/null, 舞女/null, 在/null, 暗夜/null, 中/null, 扔/null, 骰子/null, 的/null, 情景/null, ,/null, 寄托/null, 了/null, 作者/null, 对/null, 庶民/null, 生活区/null, 的/null, 情感/null]
    [这个/null, 像/null, 是/null, 真的/null, [/null, 委屈/null, ]/null, 前面/null, 那个/null, 打扮/null, 太/null, 江户/null, 了/null, ，/null, 一点/null, 不/null, 上/null, 品/null, ./null, ./null, ./null, @/null, h/null, a/null, n/null, k/null, c/null, s/null]
    [鼎泰丰/null, 的/null, 小笼/null, 一点/null, 味道/null, 也/null, 没有/null, ./null, ./null, ./null, 每样/null, 都/null, 淡淡的/null, ./null, ./null, ./null, 淡淡的/null, ，/null, 哪/null, 有/null, 食堂/null, 2/null, A/null, 的/null, 好/null, 次/null]
    [克里斯蒂娜/null, ·/null, 克/null, 罗/null, 尔/null, 说/null, ：/null, 不/null, ，/null, 我/null, 不是/null, 虎妈/null, 。/null, 我/null, 全家/null, 都/null, 热爱/null, 音乐/null, ，/null, 我/null, 也/null, 鼓励/null, 他们/null, 这么/null, 做/null, 。/null]
    [今日/null, A/null, P/null, P/null, S/null, ：/null, S/null, a/null, g/null, o/null,  /null, M/null, i/null, n/null, i/null,  /null, T/null, o/null, o/null, l/null, b/null, o/null, x/null, 培养/null, 孩子/null, 动手/null, 能力/null]
    [财政部/null, 副部长/null, 王/null, 保安/null, 调任/null, 国家/null, 统计局/null, 党组/null, 书记/null]
    [2/null, ./null, 3/null, 4/null, 米/null, 男子/null, 娶/null, 1/null, ./null, 5/null, 3/null, 米/null, 女/null, 粉丝/null,  /null, 称/null, 夫妻/null, 生活/null, 没问题/null]
    [你/null, 看过/null, 穆/null, 赫/null, 兰/null, 道/null, 吗/null]
    [国办/null, 发布/null, 网络/null, 提速/null, 降/null, 费/null, 十/null, 四/null, 条/null, 指导/null, 意见/null,  /null, 鼓励/null, 流量/null, 不/null, 清零/null]
    [乐/null, 视/null, 超级/null, 手机/null, 能否/null, 承载/null, 贾/null, 布/null, 斯/null, 的/null, 生态/null, 梦/null]
    DATnewSegment 分词速度：652231.33字每秒


## viterbi 维特比分词


```python
ViterbiNewSegment = HanLP.newSegment("viterbi")

for sentence in sentence_array:
    term_list = ViterbiNewSegment.seg(sentence)
    print(term_list)
    
import time
pressure = 10000

start = time.time()
for i in range(pressure):
    ViterbiNewSegment.seg(text)
cost_time = time.time() - start
print("ViterbiNewSegment 分词速度：%.2f字每秒" % (len(text) * pressure / cost_time))
```

    [HanLP/nx, 是/vshi, 由/p, 一系列/b, 模型/n, 与/cc, 算法/n, 组成/v, 的/ude1, Java/nx, 工具包/n, ，/w, 目标/n, 是/vshi, 普及/v, 自然语言处理/nz, 在/p, 生产/vn, 环境/n, 中的/v, 应用/vn, 。/w]
    [鐵/n, 桿/w, 部/q, 隊/n, 憤/w, 怒/vg, 情/n, 緒/n, 集/q, 結/n,  /w, 馬/nz, 英/b, 九/m, 腹/ng, 背/v, 受/v, 敵/w]
    [馬/nz, 英/b, 九/m, 回/v, 應/w, 連/n, 勝/nz, 文/ng, “/w, 丐/n, 幫/nz, 說/v, ”/w, ：/w, 稱/v, 黨/w, 內/nz, 同志/n, 談話/n, 應/w, 謹/n, 慎/ag]
    [高锰酸钾/nf, ，/w, 强氧化剂/gc, ，/w, 紫红色/n, 晶体/n, ，/w, 可/v, 溶于/v, 水/n, ，/w, 遇/v, 乙醇/n, 即/v, 被/pbei, 还原/vi, 。/w, 常用/a, 作/v, 消毒剂/n, 、/w, 水/n, 净化/vn, 剂/q, 、/w, 氧化剂/n, 、/w, 漂白剂/n, 、/w, 毒气/n, 吸收剂/nz, 、/w, 二氧化碳/n, 精制/v, 剂/q, 等/udeng, 。/w]
    [《/w, 夜晚/n, 的/ude1, 骰子/n, 》/w, 通过/p, 描述/v, 浅草/nz, 的/ude1, 舞女/n, 在/p, 暗夜/nz, 中/f, 扔/v, 骰子/n, 的/ude1, 情景/n, ,/w, 寄托/v, 了/ule, 作者/nnt, 对/p, 庶民/n, 生活区/n, 的/ude1, 情感/n]
    [这个/rz, 像/v, 是/vshi, 真的/d, [/w, 委屈/a, ]/w, 前面/f, 那个/rz, 打扮/v, 太/d, 江户/ns, 了/ule, ，/w, 一点/m, 不/d, 上品/n, .../w, @hankcs/nx]
    [鼎泰丰/nz, 的/ude1, 小笼/nf, 一点/m, 味道/n, 也/d, 没有/v, .../w, 每样/nz, 都/d, 淡淡的/z, .../w, 淡淡的/z, ，/w, 哪/ry, 有/vyou, 食堂/n, 2/m, A/nx, 的/ude1, 好/a, 次/qv]
    [克里斯蒂娜/nr, ·/w, 克/q, 罗尔/nr, 说/v, ：/w, 不/d, ，/w, 我/rr, 不是/c, 虎妈/nz, 。/w, 我/rr, 全家/n, 都/d, 热爱/v, 音乐/n, ，/w, 我/rr, 也/d, 鼓励/v, 他们/rr, 这么/rz, 做/v, 。/w]
    [今日/t, APPS/nx, ：/w, Sago/nx,  /w, Mini/nx,  /w, Toolbox/nx, 培养/v, 孩子/n, 动手/vi, 能力/n]
    [财政部/nt, 副部长/nnt, 王保安/nr, 调任/v, 国家/n, 统计局/nis, 党组/nis, 书记/nnt]
    [2.34/m, 米/q, 男子/n, 娶/v, 1.53/m, 米/q, 女/b, 粉丝/nf,  /w, 称/v, 夫妻/n, 生活/vn, 没问题/nz]
    [你/rr, 看过/v, 穆赫/nr, 兰道/nr, 吗/y]
    [国办/vn, 发布/v, 网络/n, 提速/vn, 降/v, 费/n, 十四/m, 条/q, 指导/vn, 意见/n,  /w, 鼓励/v, 流量/n, 不/d, 清零/nz]
    [乐/a, 视/vg, 超级/b, 手机/n, 能否/v, 承载/v, 贾布斯/nr, 的/ude1, 生态/n, 梦/n]
    ViterbiNewSegment 分词速度：233386.22字每秒


## 感知器分词


```python
PerceptronNewSegment = HanLP.newSegment("perceptron")

for sentence in sentence_array:
    term_list = PerceptronNewSegment.seg(sentence)
    print(term_list)
    
import time
pressure = 10000

start = time.time()
for i in range(pressure):
    PerceptronNewSegment.seg(text)
cost_time = time.time() - start
print("PerceptronNewSegment 分词速度：%.2f字每秒" % (len(text) * pressure / cost_time))
```

    [HanLP/nx, 是/v, 由/p, 一系列/n, 模型/n, 与/c, 算法/n, 组成/v, 的/u, Java/nx, 工具包/n, ，/w, 目标/n, 是/v, 普及/v, 自然语言处理/v, 在/p, 生产环境/l, 中的/u, 应用/vn, 。/w]
    [鐵桿/n, 部隊/n, 憤怒/vn, 情緒/n, 集結 馬英九/vn, 腹背受敵/Ng]
    [馬英九/nr, 回應/v, 連勝文/n, “/w, 丐幫/n, 說/v, ”/w, ：/v, 稱/v, 黨內/s, 同志/n, 談話/v, 應/v, 謹慎/a]
    [高锰酸钾/nr, ，/w, 强/a, 氧化剂/n, ，/w, 紫红色/n, 晶体/n, ，/w, 可/v, 溶于/v, 水/n, ，/w, 遇/v, 乙醇/n, 即/d, 被/p, 还原/v, 。/w, 常用/d, 作/v, 消毒剂/n, 、/w, 水/n, 净化剂/n, 、/w, 氧化剂/n, 、/w, 漂白剂/n, 、/w, 毒气/n, 吸收/v, 剂/n, 、/w, 二氧化碳/n, 精/a, 制剂/n, 等/u, 。/w]
    [《/w, 夜晚/t, 的/u, 骰子/n, 》/w, 通过/p, 描述/v, 浅草/n, 的/u, 舞女/n, 在/p, 暗夜/t, 中/f, 扔/v, 骰子/n, 的/u, 情景/n, ,/w, 寄托/v, 了/u, 作者/n, 对/p, 庶民/n, 生活区/n, 的/u, 情感/n]
    [这个/r, 像/n, 是/v, 真的/d, [/w, 委屈/vn, ]/w, 前面/f, 那个/r, 打扮/v, 太/d, 江户/a, 了/y, ，/w, 一点/m, 不/d, 上品/v, ...@hankcs/Ng]
    [鼎泰丰/nr, 的/u, 小笼/n, 一点/m, 味道/n, 也/d, 没有/v, ...每样/r, 都/d, 淡淡/z, 的/u, ...淡淡/z, 的/u, ，/w, 哪/r, 有/v, 食堂/n, 2/m, A/nx, 的/u, 好/a, 次/Bg]
    [克里斯蒂娜·克罗尔/nr, 说/v, ：/v, 不/d, ，/w, 我/r, 不是/c, 虎妈/n, 。/w, 我/r, 全家/n, 都/d, 热爱/v, 音乐/n, ，/w, 我/r, 也/d, 鼓励/v, 他们/r, 这么/r, 做/v, 。/w]
    [今日/t, APPS/nx, ：/v, Sago Mini Toolbox/nx, 培养/v, 孩子/n, 动手/v, 能力/n]
    [财政部/nt, 副部长/n, 王保安/nr, 调任/v, 国家统计局/nt, 党组/n, 书记/n]
    [2.34/m, 米/q, 男子/n, 娶/v, 1.53/m, 米/q, 女/b, 粉丝 称夫妻/n, 生活/vn, 没问题/n]
    [你/r, 看过/v, 穆赫兰道/n, 吗/y]
    [国办/j, 发布/vn, 网络/n, 提速/v, 降费/n, 十四/m, 条/q, 指导/v, 意见 鼓励/v, 流量/n, 不/d, 清零/a]
    [乐/a, 视/Vg, 超级/b, 手机/n, 能否/v, 承载/v, 贾布斯/nr, 的/u, 生态/n, 梦/n]
    PerceptronNewSegment 分词速度：58435.77字每秒


## 特殊的：HMM2 分词

hmm2 不同于CRF,一开始在废除之后并没有替代的函数，但是源代码中依旧保留了这接口，这是一个BUG，不过在我写这个用户指南期间，我已经报告了这个错误。在文章发表前，原作者已经修复了该问题。但是这事很有意思，所以还是留下了记录（当然现在的报错已经不一样了）。同时在这里也希望如果之后有人在发现错误之后，能够报告错误为开源社区作出贡献，而不仅仅是做一个受益者。


```python
HMM2newSegment = HanLP.newSegment("hmm2")

for sentence in sentence_array:
    term_list = HMM2newSegment.seg(sentence)
    print(term_list)
    
import time
pressure = 10000

start = time.time()
for i in range(pressure):
    HMM2newSegment.seg(text)
cost_time = time.time() - start
print("HMM2newSegment 分词速度：%.2f字每秒" % (len(text) * pressure / cost_time))
```


    ---------------------------------------------------------------------------

    java.lang.IllegalArgumentExceptionPyRaisableTraceback (most recent call last)

    <ipython-input-15-3cf032a3824c> in <module>()
    ----> 1 HMM2newSegment = HanLP.newSegment("hmm2")
          2 
          3 for sentence in sentence_array:
          4     term_list = HMM2newSegment.seg(sentence)
          5     print(term_list)


    java.lang.IllegalArgumentExceptionPyRaisable: java.lang.IllegalArgumentException: 发生了异常：java.lang.IllegalArgumentException: HMM分词模型[ /home/fonttian/anaconda3/lib/python3.6/site-packages/pyhanlp/static/data/model/segment/HMMSegmentModel.bin ]不存在
    	at com.hankcs.hanlp.seg.HMM.HMMSegment.<init>(HMMSegment.java:51)
    	at com.hankcs.hanlp.seg.HMM.HMMSegment.<init>(HMMSegment.java:36)
    	at com.hankcs.hanlp.HanLP.newSegment(HanLP.java:680)


