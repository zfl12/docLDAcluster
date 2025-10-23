import jieba
import jieba.posseg as psg
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pyLDAvis
import pyLDAvis.sklearn
#定义函数，加载txt文件
def readtxt(filepath,encoding='utf-8'):
    words = [line.strip() for line in open(filepath, mode='r',encoding=encoding).readlines()]
    return words
#调用函数
text = readtxt('./text.txt')#一个文章摘要形成列表中的一个元素


# 定义分词函数
def cut_word(text):
    # 加载用户自定义词典
    jieba.load_userdict("./user_dict.txt")
    # 加载停用词表
    stopwords = readtxt('./stopwords.txt', encoding='gbk')
    sentence = ""
    checkarr = ['n']
    for word, flag in psg.lcut(text):
        if (flag in checkarr) and (word not in stopwords) and (len(word) > 1):
            sentence = sentence + word + " "
    return sentence


# 分词
segged_words = [cut_word(x) for x in text]
n_features = 1000# 指定特征关键词提取最大值
cv = CountVectorizer(strip_accents = 'unicode',#将使用unicode编码在预处理步骤去除raw document中的重音符号
                                max_features=n_features,
                                max_df = 0.5,# 阈值如果某个词的document frequence大于max_df，不当作关键词
                                min_df = 3# 如果某个词的document frequence小于min_df，则这个词不会被当作关键词
                               )
tf = cv.fit_transform(segged_words)
#查看构建的词典
print(cv.vocabulary_)
#说明：'研究成果': 67,表示'研究成果'这个词在词典中的索引为67
#查看词典大小
print(len(cv.vocabulary_))
#98
#查看抽取出的特征词
print(cv.get_feature_names())
#查看抽取出的特征词个数
print(len(cv.get_feature_names()))
#98
#查看每个特征在单个文摘中的词频
print(tf)
#结果如下：
#说明：
#以(0, 67)  1为例：其中，0表示第0个列表元素，即第一篇论文摘要。67表示该词在构建的词典中的索引为67，1为该词在该篇论文中的词频。
#查看全部文摘向量化表示的结果
print(tf.toarray())
#说明：每一行为一篇文摘。共抽取出98个特征词，因此共98列，每一列上的数字表示该文章在该特征词的词频
#第一篇文摘的词向量结果
print(tf.toarray()[0])


#第一篇文摘的词向量长度
print(len(tf.toarray()[0]))
#98
#再次说明，此处基于词频的方法对每个文摘向量化的本质，是根据全部文摘抽取出特征词，然后计算每个特征词在单个文摘上的词频。

print(tf.toarray().sum(axis=0))
# （1）获取高频词的索引
fre = tf.toarray().sum(axis=0)
index_lst = []
for i in range(len(fre)):
    if fre[i] > 10:  # 词频大于10的定义为高频词
        index_lst.append(i)

# （2）对词典按词频升序排序
voca = list(cv.vocabulary_.items())
sorted_voca = sorted(voca, key=lambda x: x[1], reverse=False)

# （3）提取高频词
high_fre_vaca = []
for i in sorted_voca:
    if i[1] in index_lst:
        high_fre_vaca.append(i[0])

#模型初始化
k= 3# 人为指定划分的主题数k
lda = LatentDirichletAllocation(n_components=k,
                                max_iter=50,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
time_start=time.time()
ldamodel = lda.fit_transform(tf)
time_end=time.time()
print('time cost',time_end-time_start,'s')
#time cost 1.294196605682373 s
proba = np.array(ldamodel)
print('每个文摘属于各个主题的概率:\n', proba)
#说明：每一行表示一个文摘，每列表示该文摘属于该主题的概率。共50篇文摘，故有50行，共3个主题，因此有3列。

# 构建一个零矩阵
zero_matrix = np.zeros([proba.shape[0]])
# 对比所属两个概率的大小，确定属于的类别
max_proba = np.argmax(proba, axis=1) # 返回沿轴axis最大值的索引，axis=1代表行；最大索引即表示最可能表示的数字是多少
print('每个文档所属类别：', max_proba)

weight_matrix = lda.components_
weight_matrix
#说明：3个主题，因此，形成了一个[[],[],[]]的嵌套列表，第一个子列表属于主题0，第二个子列表属于主题1，……每个子列表均有98个特征词，数字表示每个特征词属于该主题的权重
#len(weight_matrix[0])#共98个

weight_matrix = lda.components_
tf_feature_names = cv.get_feature_names()
id = 0
for weights in weight_matrix:
    dicts = [(name, weight) for name, weight in zip(tf_feature_names, weights)]
    dicts = sorted(dicts, key=lambda x: x[1], reverse=True)#根据特征词的权重降序排列
    dicts = [word for word in dicts if word[1] > 0.6]# 打印权重值大于0.6的主题词
    dicts = dicts[:5]# 打印每个主题前5个主题词
    print('主题%d:' % (id), dicts)
    id += 1