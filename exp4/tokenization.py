import jieba
import jieba.analyse
import os
import re
from gensim.models import word2vec

folder = r'novels\save'
merge_file = r'agg.txt'
model_sp = r'novels.model'
# jieba.analyse.set_stop_words('stop_words.txt')
def load_stopwords():
    with open('stop_words.txt', encoding='utf-8') as f:
        stop_words = [x.strip('\n') for x in f.readlines()]
        stop_words = set(stop_words)
    return stop_words

def tokenize_novels():
    stop_words = load_stopwords()
    mf = open(merge_file, mode='w', encoding='utf-8')
    files = [os.path.join(folder, fname) for fname in os.listdir(folder)]
    for file in files:
        f = open(file, mode='r', encoding='utf-8')
        for line in f.readlines():
            nopunc = re.sub(r'[^\u4e00-\u9fa5]', '', line)
            cut= jieba.lcut(nopunc, cut_all=False)
            cutted = set(cut)
            wline = ' '.join(sorted(cutted-stop_words, key=cut.index))+'\r\n'
            mf.write(wline)
    mf.close()

def train():
    sentences = word2vec.Text8Corpus(merge_file)
    model = word2vec.Word2Vec(sentences, size=200)
    model.save(model_sp)
    return model

if __name__ == '__main__':
    # model = train()
    model = word2vec.Word2Vec.load(model_sp)
    query = '金丹'
    print('most similar words to "{}" learned from this corpus are :\n{}'
          .format(query,model.most_similar(query,topn=3)))

