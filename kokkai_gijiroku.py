
##1国会議事録データの取得
#データのスクレイピング
import urllib#ネットのurlから取得するモジュール
import untangle
import urllib.parse
if __name__ == '__main__':
    start='1'#発言の通し番号
    while start!=None:
        keyword = ''
        startdate='2008-01-01'
        enddate= '2008-01-31'
        meeting='本会議'
        #urllib.parse.quoteが日本語をコーディングしてくれる
        url = 'http://kokkai.ndl.go.jp/api/1.0/speech?'+urllib.parse.quote('startRecord='+ start
        + '&maximumRecords=100&speaker='+ keyword
        + '&nameOfMeeting='+ meeting
        + '&from=' + startdate
        + '&until='+ enddate)
        #Get信号のリクエストの検索結果（XML）
        obj = untangle.parse(url)

        for record in obj.data.records.record:
            speechrecord = record.recordData.speechRecord
            print(speechrecord.date.cdata,
                speechrecord.speech.cdata)

            file=open('data_2008_01.txt','a')
            file.write(speechrecord.speech.cdata)
            file.close()
            #一度に１００件しか帰ってこないので、開始位置を変更して繰り返しGET関数を送信
        start=obj.data.nextRecordPosition.cdata

#これを繰り返して08年から18年までのデータを取得


##2議事録データを発言者ごとに文書に切り分け、形態素解析を行う
#必要なモジュールのインポート
from urllib import request 
import logging
from pathlib import Path
import numpy as np
import re
import MeCab
mecab = MeCab.Tagger("-Ochasen -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd/")
import random
from gensim import corpora, models

##前処理

#日本語ストップワードをネットから取得
res = request.urlopen("http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/Japanese.txt")
stopwords = [line.decode("utf-8").strip() for line in res]
#英語ストップワードをネットから取得
res = request.urlopen("http://svn.sourceforge.jp/svnroot/slothlib/CSharp/Version1/SlothLib/NLP/Filter/StopWord/word/English.txt")
stopwords += [line.decode("utf-8").strip() for line in res]
#国会議事録にあるいらない言葉消す
mydict = ['安倍晋三','ん','国民','我が国','内閣総理大臣','日本','重要','の','平成','お尋ね','今後','昨年','必要','実現','改革','強化','政府','環境','対策','制度','経済','世界']
stopwords = stopwords + mydict

class Tokenizer:
    def __init__(self, stopwords, parser=None, include_pos=None, exclude_posdetail=None, exclude_reg=None):
    
        self.stopwords = stopwords
        self.include_pos = include_pos if include_pos else  ["名詞"]#名詞のみ
        self.exclude_posdetail = exclude_posdetail if exclude_posdetail else ["接尾", "数"]
        self.exclude_reg = exclude_reg if exclude_reg else r"$^"  # no matching reg
        if parser:
            self.parser = parser
        else:
            mecab = MeCab.Tagger("-Ochasen -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd/")
            self.parser = mecab.parse
            

    def tokenize(self, text, show_pos=False):
        text = re.sub(r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+", "", text)    
        text = re.sub(r"\"?([-a-zA-Z0-9.`?{}]+\.jp)\"?" ,"", text)  
        text = text.lower()
        l = [line.split("\t") for line in self.parser(text).split("\n")]
        res = [
            i[2] if not show_pos else (i[2],i[3]) for i in l 
                if len(i) >=4 # has POS.
                    and i[3].split("-")[0] in self.include_pos
                    and i[3].split("-")[1] not in self.exclude_posdetail
                    and not re.search(r"(-|−)\d", i[2])
                    and not re.search(self.exclude_reg, i[2])
                    and i[2] not in self.stopwords          
            ]
        return res
t = Tokenizer(stopwords + ["…。"] , mecab.parse, exclude_reg=r"\d(年|月|日)")


#各発言を文書としてデータを切って前処理を行う
texts=[]
docs=[]
dict_2018 = {}#辞書型
for num in ['1','2','3','4','5','6','9','10']:#7こ
    dict_2018[num]= open(f'data_2018_{num}.txt')
    data1 = dict_2018[num].read()#全てのデータを返す
    dict_2018[num].close()
    #ここでテキストを切って文書化
    sentences = data1.split('○')
    #文書ごとに前処理してリストに入れる
    for sentence in sentences:
        texts.append(sentence)
        docs.append(t.tokenize(sentence))

dict_2017 = {}#辞書型
for num in ['1','2','3','4','5','6','9','11','12']:#9こ
    dict_2017[num]= open(f'data_2017_{num}.txt')
    data1 = dict_2017[num].read()#全てのデータを返す
    dict_2017[num].close()
#     ここでテキストを切って文書化
    sentences = data1.split('○')
    #文書ごとに前処理してリストに入れる
    for sentence in sentences:
        texts.append(sentence)
        docs.append(t.tokenize(sentence))

dict_2016 = {}#辞書型
for num in ['1','2','3','4','5','6','8','9','10','11','12']:#11こ
    dict_2016[num]= open(f'data_2016_{num}.txt')
    data1 = dict_2016[num].read()#全てのデータを返す
    dict_2016[num].close()
#     ここでテキストを切って文書化
    sentences = data1.split('○')
    #文書ごとに前処理してリストに入れる
    for sentence in sentences:
        texts.append(sentence)
        docs.append(t.tokenize(sentence))

        
dict_2015 = {}#辞書型
for num in ['1','2','3','4','5','6','7','8','9']:#
    dict_2015[num]= open(f'data_2015_{num}.txt')
    data1 = dict_2015[num].read()#全てのデータを返す
    dict_2015[num].close()
    #ここでテキストを切って文書化
    sentences = data1.split('○')
    #文書ごとに前処理してリストに入れる
    for sentence in sentences:
        texts.append(sentence)
        docs.append(t.tokenize(sentence))
        


dict_2014 = {}#辞書型
for num in ['1','2','3','4','5','6','9','10','11','12']:#
    dict_2014[num]= open(f'data_2014_{num}.txt')
    data1 = dict_2014[num].read()#全てのデータを返す
    dict_2014[num].close()
    #ここでテキストを切って文書化
    sentences = data1.split('○')
    #文書ごとに前処理してリストに入れる
    for sentence in sentences:
        texts.append(sentence)
        docs.append(t.tokenize(sentence))

dict_2013 = {}#辞書型
for num in ['1','2','3','4','5','6','8','10','11','12']:#
    dict_2013[num]= open(f'data_2013_{num}.txt')
    data1 = dict_2013[num].read()#全てのデータを返す
    dict_2013[num].close()
    #ここでテキストを切って文書化
    sentences = data1.split('○')
    #文書ごとに前処理してリストに入れる
    for sentence in sentences:
        texts.append(sentence)
        docs.append(t.tokenize(sentence))

dict_2012 = {}#辞書型
for num in ['1','2','3','4','5','6','7','8','9','10','11','12']:#
    dict_2012[num]= open(f'data_2012_{num}.txt')
    data1 = dict_2012[num].read()#全てのデータを返す
    dict_2012[num].close()
    #ここでテキストを切って文書化
    sentences = data1.split('○')
    #文書ごとに前処理してリストに入れる
    for sentence in sentences:
        texts.append(sentence)
        docs.append(t.tokenize(sentence))

dict_2011 = {}#辞書型
for num in ['1','2','3','4','5','6','7','8','9','10','11','12']:#
    dict_2011[num]= open(f'data_2011_{num}.txt')
    data1 = dict_2011[num].read()#全てのデータを返す
    dict_2011[num].close()
    #ここでテキストを切って文書化
    sentences = data1.split('○')
    #文書ごとに前処理してリストに入れる
    for sentence in sentences:
        texts.append(sentence)
        docs.append(t.tokenize(sentence))
        
dict_2010 = {}#辞書型
for num in ['1','2','3','4','5','6','7','8','10','11','12']:#
    dict_2010[num]= open(f'data_2010_{num}.txt')
    data1 = dict_2010[num].read()#全てのデータを返す
    dict_2010[num].close()
    #ここでテキストを切って文書化
    sentences = data1.split('○')
    #文書ごとに前処理してリストに入れる
    for sentence in sentences:
        texts.append(sentence)
        docs.append(t.tokenize(sentence))

dict_2009 = {}#辞書型
for num in ['1','2','3','4','6','7','9','10','11','12']:#
    dict_2009[num]= open(f'data_2009_{num}.txt')
    data1 = dict_2009[num].read()#全てのデータを返す
    dict_2009[num].close()
    #ここでテキストを切って文書化
    sentences = data1.split('○')
    #文書ごとに前処理してリストに入れる
    for sentence in sentences:
        texts.append(sentence)
        docs.append(t.tokenize(sentence))
        
dict_2008 = {}#辞書型
for num in ['1','2','3','4','5','6','9','10','11','12']:#
    dict_2008[num]= open(f'data_2008_{num}.txt')
    data1 = dict_2008[num].read()#全てのデータを返す
    dict_2008[num].close()
    #ここでテキストを切って文書化
    sentences = data1.split('○')
    #文書ごとに前処理してリストに入れる
    for sentence in sentences:
        texts.append(sentence)
        docs.append(t.tokenize(sentence))
        
print(len(texts), len(docs))
print( [len(v) for v in docs][:5])



##3コーパスの作成



#辞書の設定
dictionary = corpora.Dictionary(docs)
docs_words = dictionary.token2id#辞書型
print(len(docs_words))

#フィルタリング
dictionary.filter_extremes(no_below=3,no_above=0.5)
#使われている文章がno_belowより少ない単語を無視、no_aboveの割合以上に出てくる単語を無視
dictionary.compactify()#IDを振り直してコンパクトにする#一つのテキストでやると全部消えるから注意
docs_words_compact = dictionary.token2id
print(len(docs_words_compact))

corpus = [dictionary.doc2bow(w) for w in docs]



##4LDAで学習を行う



import logging
import random
from gensim import models
#モデルの学習
#時間がかかります
#test,trainに分けて学習
test_size = int(len(corpus) * 0.1)
test_corpus = corpus[:test_size]#１割
train_corpus = corpus[test_size:]#9割

#学習状況を確認するためのlogの設定
logging.basicConfig(format='%(message)s', level=logging.INFO)

#トピック数50に設定して学習
lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=50, passes=10)

#Nはのべの出現単語数(test)
N = sum(count for doc in test_corpus for id, count in doc)
print("N: ",N)

#perplexityで性能評価
perplexity = np.exp2(-lda.log_perplexity(test_corpus))
print("perplexity:", perplexity)

# 各トピックの要素の表示
topic50 = []
for topic_ in lda.show_topics(num_topics=50, num_words=100, formatted=False):
    topic50.append([token_[0] for token_ in topic_[1]])

topic50 = pd.DataFrame(topic50)
#print(topic50)
topic50.to_csv(('topic08-18.csv').__str__(), index=False, encoding='utf-8')


#文書ごとのトピック分類結果を得る
topics = [lda[c] for c in corpus]

def sort_(x):
    return sorted(x, key=lambda x_:x_[1], reverse=True)
#トピックを取得
num_topics = lda.get_topics().shape[0]
target_doc_id =539
print(texts[target_doc_id])
print(sort_(topics[target_doc_id]))



##5トピック文書数の年ごとの変化を見る



#年ごとにトピック文書を格納するリストの作成
for num in range(50):
    exec('topic{}_text =[]'.format(num))
    exec('topic{}_text_number=[]'.format(num))
    for j in range(2008,2019):
        exec('topic{}_text_{} =[]'.format(num, j))
#トピックの指定
for i in range(34205):
    for j in range(len(topics[i])):
        if sort_(topics[i])[j][0]==38:
            if sort_(topics[i])[j][1]>=0.20:
                topic38_text.append(texts[i])
                topic38_text_number.append(i)
#該当する文書の数の確認
print(len(topic38_text_number), len(topic38_text))

for i in range(len(topic38_text_number)):
    if topic38_text_number[i]<=2876:
        topic38_text_2018.append(topic38_text_number[i])
    elif 2876< topic38_text_number[i]<=5694:
        topic38_text_2017.append(topic38_text_number[i])
    elif 5694< topic38_text_number[i]<=9005:
        topic38_text_2016.append(topic38_text_number[i])
    elif 9005< topic38_text_number[i]<=11958:
        topic38_text_2015.append(topic38_text_number[i])
    elif 11958< topic38_text_number[i]<=15403:
        topic38_text_2014.append(topic38_text_number[i])
    elif 15403< topic38_text_number[i]<=19687:
        topic38_text_2013.append(topic38_text_number[i])
    elif 19687< topic38_text_number[i]<=22293:
        topic38_text_2012.append(topic38_text_number[i])
    elif 22293< topic38_text_number[i]<=25584:
        topic38_text_2011.append(topic38_text_number[i])
    elif 25584< topic38_text_number[i]<=28649:
        topic38_text_2010.append(topic38_text_number[i])
    elif 28649< topic38_text_number[i]<=31419:
        topic38_text_2009.append(topic38_text_number[i])
    elif 31419< topic38_text_number[i]<=34205:
        topic38_text_2008.append(topic38_text_number[i])

topic38_time=[
              len(topic38_text_2008),
              len(topic38_text_2009),
              len(topic38_text_2010),
              len(topic38_text_2011),
              len(topic38_text_2012),
              len(topic38_text_2013),
              len(topic38_text_2014),
              len(topic38_text_2015),
              len(topic38_text_2016),
              len(topic38_text_2017),
              len(topic38_text_2018)]
topic38_time

#これの繰り返し。
for i in range(len(topic25_text_number)):
    if topic25_text_number[i]<=2876:
        topic25_text_2018.append(topic25_text_number[i])
    elif 2876< topic25_text_number[i]<=5694:
        topic25_text_2017.append(topic25_text_number[i])
    elif 5694< topic25_text_number[i]<=9005:
        topic25_text_2016.append(topic25_text_number[i])
    elif 9005< topic25_text_number[i]<=11958:
        topic25_text_2015.append(topic25_text_number[i])
    elif 11958< topic25_text_number[i]<=15403:
        topic25_text_2014.append(topic25_text_number[i])
    elif 15403< topic25_text_number[i]<=19687:
        topic25_text_2013.append(topic25_text_number[i])
    elif 19687< topic25_text_number[i]<=22293:
        topic25_text_2012.append(topic25_text_number[i])
    elif 22293< topic25_text_number[i]<=25584:
        topic25_text_2011.append(topic25_text_number[i])
    elif 25584< topic25_text_number[i]<=28649:
        topic25_text_2010.append(topic25_text_number[i])
    elif 28649< topic25_text_number[i]<=31419:
        topic25_text_2009.append(topic25_text_number[i])
    elif 31419< topic25_text_number[i]<=34205:
        topic25_text_2008.append(topic25_text_number[i])

topic25_time=[
              len(topic25_text_2008),
              len(topic25_text_2009),
              len(topic25_text_2010),
              len(topic25_text_2011),
              len(topic25_text_2012),
              len(topic25_text_2013),
              len(topic25_text_2014),
              len(topic25_text_2015),
              len(topic25_text_2016),
              len(topic25_text_2017),
              len(topic25_text_2018)]
topic25_time

#トピックの指定
for i in range(34205):
    for j in range(len(topics[i])):
        if sort_(topics[i])[j][0]==29:
            if sort_(topics[i])[j][1]>=0.20:
                topic29_text.append(texts[i])
                topic29_text_number.append(i)
#該当する文書の数の確認
print(len(topic29_text_number), len(topic29_text))

for i in range(len(topic29_text_number)):
    if topic29_text_number[i]<=2876:
        topic29_text_2018.append(topic29_text_number[i])
    elif 2876< topic29_text_number[i]<=5694:
        topic29_text_2017.append(topic29_text_number[i])
    elif 5694< topic29_text_number[i]<=9005:
        topic29_text_2016.append(topic29_text_number[i])
    elif 9005< topic29_text_number[i]<=11958:
        topic29_text_2015.append(topic29_text_number[i])
    elif 11958< topic29_text_number[i]<=15403:
        topic29_text_2014.append(topic29_text_number[i])
    elif 15403< topic29_text_number[i]<=19687:
        topic29_text_2013.append(topic29_text_number[i])
    elif 19687< topic29_text_number[i]<=22293:
        topic29_text_2012.append(topic29_text_number[i])
    elif 22293< topic29_text_number[i]<=25584:
        topic29_text_2011.append(topic29_text_number[i])
    elif 25584< topic29_text_number[i]<=28649:
        topic29_text_2010.append(topic29_text_number[i])
    elif 28649< topic29_text_number[i]<=31419:
        topic29_text_2009.append(topic29_text_number[i])
    elif 31419< topic29_text_number[i]<=34205:
        topic29_text_2008.append(topic29_text_number[i])

topic29_time=[
              len(topic29_text_2008),
              len(topic29_text_2009),
              len(topic29_text_2010),
              len(topic29_text_2011),
              len(topic29_text_2012),
              len(topic29_text_2013),
              len(topic29_text_2014),
              len(topic29_text_2015),
              len(topic29_text_2016),
              len(topic29_text_2017),
              len(topic29_text_2018)]
topic29_time


#topic25:教育,27:外交 29農業, 38:自衛隊,39:tpp,  48：金融
topic25_time,topic27_time,topic29_time,topic38_time,topic39_time, topic48_time,

#可視化
x=np.arange(2008,2019,1)
plt.plot(x,topic2_time, label='2Earthquake')
plt.plot(x,topic25_time, label='25Education')
plt.plot(x, topic27_time, label='27Diplomacy')
plt.plot(x, topic29_time,  label='29Agriculture')
plt.plot(x, topic38_time, label='38army')
plt.plot(x, topic39_time, label='39tpp')
plt.plot(x, topic48_time,  label='48Finance')
plt.title('topic time series')
plt.ylabel('docs_number')
plt.legend(loc='best')
plt.show()

