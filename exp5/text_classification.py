import re
import jieba
from gensim import corpora,models
from Bayes import NaiveBayes
# import Bayes
import numpy as np
class spam_detection(NaiveBayes):
    def __init__(self):
        super().__init__(continuous_attr=[],
                         use_validation=False, train_ratio=1,
                         use_pca=False, n_components=120)
        self.stop_words_path = r'stop_words.txt'
        self.punctuation = re.compile("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+")

        self.stop_words = spam_detection.load_stopwords(self.stop_words_path)

        self.one_hot = lambda x:[1 if k in x else 0 for k in self.vocabulary.values()]


    def _read_data(self, file, train=True):
        if train:
            mails,spams = file
            self.prepare_onehot(mails, spams)
            self.all_samples = np.array(self.positive_samples+self.negative_samples)
            self.all_raw_labels = np.array(self.positive_labels+self.negative_labels)
            self.cls = ['spam', 'normal']

            self.n_classes = len(self.cls)
            self.n_all_samples = len(self.all_samples)
            self.n_attribute = len(self.all_samples[0])
        else: # testing
            tokenized,lines = self.tokenize_mails(file)
            self.all_samples = np.array(lines)
            self.samples = np.array([self.one_hot(line) for line in tokenized])
            self.all_raw_labels = None


    @staticmethod
    def load_stopwords(file):
        with open(file, encoding='utf-8') as f:
            stop_words = [x.strip('\n') for x in f.readlines()]
            stop_words = set(stop_words)
        return stop_words

    def tokenize_mails(self, data_path):
        lines = []
        with open(data_path, encoding='utf-8') as f:
            for line in f.readlines():
                lines.append(self.punctuation.sub('', line))

        return [[x for x in jieba.cut(line, cut_all=True) if not x in self.stop_words] for line in lines],lines

    def prepare_onehot(self, mails_path, spams_path):
        mails_tokenized,_ = self.tokenize_mails(mails_path)
        spams_tokenized,_ = self.tokenize_mails(spams_path)

        mails_vocab = corpora.Dictionary(mails_tokenized)
        spams_vocab = corpora.Dictionary(spams_tokenized)
        mails_vocab.merge_with(spams_vocab)
        self.vocabulary = mails_vocab

        self.positive_samples = [self.one_hot(line) for line in mails_tokenized]
        self.positive_labels = [1]*len(self.positive_samples)
        self.negative_samples = [self.one_hot(line) for line in spams_tokenized]
        self.negative_labels = [0]*len(self.negative_samples)


if __name__ == '__main__':
    instance = spam_detection()
    mails_path = r'datasets\ham_100.utf8'
    spams_path = r'datasets\spam_100.utf8'
    test_path = r'datasets\test.utf8'
    instance.train(file=(mails_path,spams_path))
    instance.test(file=test_path)
