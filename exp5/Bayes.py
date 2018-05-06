# suppose a sample x = (x0,x1,...,xn)
# there is m class c0,c1,...,cm
# for unknown sample x_k, we compute p(c|x_k), where c=(c0,c1,...,cm)
# and c_pred = argmax_c{p(c|x_k}
# the bayes thm.
#     p(c_j|x_k)*p(x_k) = p(c_j, x_k)
#  =  p(x_k|c_j)*p(c_j)
# for 0<=j<=m, p(x_k) remains the same and thus can be omitted;
# p(c_j) is the prior and can be computed on the training data
# As to p(x_k|c_j):
#   according to the 'naive' nature, i.e., attributes are independent on each other
#     p(x_k|c_j) = p(x0_k,x1_k,...,xn_k|c_j)
#  =  p(x0_k|c_j)*p(x1_k|c_j)*...*p(xn_k|c_j)
#   for each attributes' conditional proba,
#   (1) attributes are discrete values, the proba is calculated as
#       count of the value divided by total count of all possible values
#   (2) attributes are continuous values, obtain a model for the
#       attribute's distribution estimated as a gaussian.
#   all of the above computation is done by the training data
import numpy as np
from scipy.stats import multivariate_normal as multi_gauss
from sklearn.decomposition import PCA
from pandas import read_excel
import logging
import abc
import threading



class Bayes(object):
    def __init__(self, continuous_attr, logger='logger.txt', dataset='unknown',
                 use_validation=False, train_ratio=None,
                 use_pca=True, n_components=120):
        np.seterr(divide='raise')
        logging.basicConfig(filename=logger, filemode='w',
                            level=logging.DEBUG, format='%(message)s')  # debug < info < warning < error < critical
        self.dataset_name = dataset
        self.continuous = continuous_attr
        self.use_split = use_validation
        self.use_pca = use_pca
        self.n_components = n_components
        self.train_ratio = train_ratio
        self.all_samples = []
        self.all_raw_labels = []
        self.cls = []
        self.n_all_samples = None
        self.n_train_samples = None
        self.n_attribute = None
        self.n_classes = None
        self.class_priors = None
        self.class_pd_priors = None  # n x m, each element is a dict, denoting for each attribute, the distribution of its values

    @abc.abstractmethod
    def _read_data(self, file, train=True):
        """
        labels should be processed into raw indices
        """
        pass

    def _split_data(self, train_ratio=0.8):
        n_train_samples = np.ceil(train_ratio*self.n_all_samples)
        self.train_indices = np.random.choice(self.n_all_samples, replace=False, size=np.int(n_train_samples))
        self.validate_indices = np.setdiff1d(np.arange(start=0,stop=self.n_all_samples), self.train_indices)


    def _compute_class_prior(self):
        self.class_priors = np.array([0] * self.n_classes, dtype=np.float32)
        for sample, label in zip(self.samples, self.raw_labels):
            self.class_priors[label] += 1
        self.class_priors /= self.n_train_samples

    @abc.abstractmethod
    def _compute_class_pd_prior(self):  # proba density
        """
        freq for discrete
        stats for continuous
        :return:
        """
        pass

    @abc.abstractmethod
    def _compute_class_probas(self, sample):
        """

        :param sample:
        :return: a list of scalars
        """
        pass

    def _predict(self, sample):
        # p(x_k|c_j)
        proba = self._compute_class_probas(sample) # overrided, using multinomial gaussian
        pred = np.argmax(proba)

        return proba, pred

    def train(self, file):
        self._read_data(file)
        if self.use_split:# split data if using validation
            self._split_data(train_ratio=0.8 if self.train_ratio is None else self.train_ratio)
            self.samples = self.all_samples[self.train_indices]
            self.raw_labels = self.all_raw_labels[self.train_indices]
            self.n_train_samples = len(self.train_indices)
        else :
            self.samples = self.all_samples
            self.raw_labels = self.all_raw_labels
            self.n_train_samples = self.n_all_samples

        self._compute_class_prior()
        self._compute_class_pd_prior() # check out self.class_priors, self.class_pd_priors
        logging.info('priors have been extracted from {} training samples'.format(self.n_train_samples))

        return self.test(file=None)

    def _prepare_labels(self):
        pass

    def test(self, file, verbose=False):
        #===== preparing data and labels ========#
        mode = None
        if file is None:# means validation after training

            if self.use_split:
                self.samples = self.all_samples[self.validate_indices]
                self.raw_labels = self.all_raw_labels[self.validate_indices]
                mode = 'validation (using split)'
            else:
                self.samples = self.all_samples
                self.raw_labels = self.all_raw_labels
                mode = 'validation (using original training set)'
        else:# provide file means testing
            # might well have loaded validation data
            self.all_samples = None
            self.all_raw_labels = None

            self._read_data(file=file, train=False)

            # self.samples = self.all_samples
            self.raw_labels = self.all_raw_labels # meaning that there will not be labels

            self._prepare_labels()  # in case for the given datasets

            mode = 'testing'

        #=========== making predicts and maps it ==========#
        tmp = [self._predict(sample) for sample in self.samples] # in validation, making predictions
        probas, raw_predicts = zip(*tmp)  # list
        mapped_predicts = [self.cls[p] for p in raw_predicts]

        #=========== logging ==============================#
        prompts = 'there are {} samples for {}'.format(len(self.samples),mode)
        logging.info('='*len(prompts))
        logging.info(prompts)
        logging.info('='*len(prompts))

        #=========== evaluating predictions ===============#
        if self.raw_labels is not None: # meaning it's validation
            # to avoid predicts converting to multidimensional array
            # the return value of each predict is a tuple
            correctness = self.raw_labels == np.array(raw_predicts)
            accuracy = np.sum(correctness)
            accuracy /= len(self.samples)
            if verbose:
                for idx, is_correct in enumerate(correctness):
                    logging.info('-------------{}-------------'.format(idx))
                    if is_correct:
                        logging.info('sample={} is predicted {}'.format(self.samples[idx], mapped_predicts[idx]))
                    else:
                        logging.warning('\n=====================================\n'
                                        + 'sample={} is predicted {}\nwrong answer, should be {}. the estimated proba is {}\n'.format(
                            self.samples[idx], mapped_predicts[idx], self.cls[self.raw_labels[idx]], probas[idx])
                                        + '=====================================\n')
            logging.info('\nthe overall accuracy is {}'.format(accuracy))
            return accuracy
        else: # meaning it's testing
            for idx, predicted in enumerate(mapped_predicts):
                logging.info('-------------{}-------------'.format(idx))
                logging.info('sample={} \nis predicted {} with estimated proba {}'.format(self.all_samples[idx], predicted,
                                                                                        probas[idx]))

            return mapped_predicts


class NaiveBayes(Bayes): # cannot make sure all float32

    def _read_data(self, file, train=True):
        # cls = []
        self.all_samples = []
        self.all_raw_labels = []
        with open(file) as f:
            for line in f.readlines():
                sample = [x.strip().strip('.') for x in line.split(',')]
                sample, label = sample[:-1], sample[-1]

                if '?' in sample or len(sample) < 2:
                    continue
                # only convert label to numbers during training
                if train:
                    if label in self.cls:
                        label = self.cls.index(label)
                    else:
                        tmp = label
                        label = len(self.cls)
                        self.cls.append(tmp)
                    self.all_raw_labels.append(label)
                else:# test
                    self.all_raw_labels.append(self.cls.index(label))

                self.all_samples.append(sample)


        self.all_samples = np.array(self.all_samples)
        self.all_raw_labels = np.array(self.all_raw_labels, dtype=np.int)
        if train:
            self.n_classes = len(self.cls)
            self.n_all_samples = len(self.all_samples)
            self.n_attribute = len(self.all_samples[0])

    def _compute_class_pd_prior(self):

        if self.continuous == 'all':
            self.continuous = []
            for i in range(self.n_attribute):
                all_cls_stds = [np.std(self.samples[c == self.raw_labels, i].astype(np.float32)) for c in
                                range(len(self.cls))]
                if not 0. in all_cls_stds:
                    self.continuous.append(i)

        tmp = [None] * (self.n_attribute * self.n_classes)
        self.class_pd_priors = np.reshape(np.array(tmp), newshape=(self.n_classes, self.n_attribute))
        # m x n x a
        # sample : n x 1 可用花式索引 [label, list(range(n_attribute)), sample]
        # self.x_c_priors = np.array([0]*(self.n_classes*self.n_attribute*self.attr_max))
        for sample, label in zip(self.samples, self.raw_labels):
            for i in range(self.n_attribute):

                if not i in self.continuous:
                    entry = self.class_pd_priors[label, i]
                    if entry is None:
                        entry = dict()
                    if entry.get(sample[i], None) is None:
                        entry[sample[i]] = 1
                    else:
                        entry[sample[i]] += 1
                    self.class_pd_priors[label, i] = entry

        for i in self.continuous:  # should make sure it's numerical value
            self.class_pd_priors[:, i] = [(np.mean(self.samples[c == self.raw_labels, i].astype(np.float32)),
                                           np.std(self.samples[c == self.raw_labels, i].astype(np.float32)))
                                          for c in range(len(self.cls))]

    def _gaussian_log_density(self, x, c, i):
        epsilon = 0 # to avoid log0
        yita, sigma = self.class_pd_priors[c, i] # yita is the average, sigma is the std-dev
        sigma += epsilon

        fraction = -(0.5*np.log(2 * np.pi)+np.log(sigma))
        exp = -(x.astype(np.float32) - yita) ** 2 / (2 * sigma ** 2)

        return fraction + exp


    def _compute_class_probas(self, sample):
        probas = []

        for i in range(self.n_classes):
            p = 0
            # for class i, compute chaining proba for each attr
            for idx, attr in enumerate(sample):
                if not idx in self.continuous:  # discrete values
                    entry = self.class_pd_priors[i, idx]  # dict
                    total = self.n_train_samples * self.class_priors[i]
                    if entry.get(attr) is None:
                        p += -np.log(total + len(list(entry.values())) + 1)
                    else:
                        p += np.log(entry[attr]) - np.log(total)
                else:
                    p += self._gaussian_log_density(attr, i, idx)  # the gaussian density for class i attr idx

            probas.append(p)

        return probas

class Bayes4Test(Bayes):
    def _read_data(self, file, train=True):
        samples_f, labels_f = file


        # process samples
        samples = read_excel(samples_f, header=None, dtype=np.float64)
        # process labels
        labels = read_excel(labels_f, header=None) if labels_f is not None else None

        self.all_samples = samples.as_matrix()
        # (len,) 1d array
        tmp = np.squeeze(labels.as_matrix(), axis=-1) if labels is not None else None  # 1,2,3
        if tmp is not None:
            # self.cls = []
            for l in tmp:
                if not l in self.cls:
                    self.cls.append(l)
                else:
                    continue

            self.all_raw_labels = np.array([self.cls.index(x) for x in tmp], dtype=np.int)
        else:
            self.all_raw_labels = None

        if train:  # only update during training
            self.n_classes = len(self.cls)  # 0,1,2
            self.n_all_samples = len(self.all_samples)
            self.n_attribute = len(self.all_samples[0])

    def test(self, file, verbose=False):
        ret = super().test(file, verbose)
        if file is None:# ret is accuracy
            return ret
        else : # ret is mapped predicts
            ret = np.expand_dims(np.array(ret), axis=-1) # len x 1
            # if file is not None:# exactly testing
            from pandas import DataFrame
            df = DataFrame(ret)
            df.to_excel('results/result_{}.xls'.format(self.dataset_name), index=False, header=False)



class Bayes4Test_PCA(Bayes4Test):
    def _read_data(self, file, train=True):
        super()._read_data(file, train) # check out self.cls,self.all_samples
        # self.tmp = self.all_samples
        if self.use_pca:
            if train:
                pca = PCA(n_components=self.n_components, whiten=True, svd_solver='full')
                pca.fit(self.all_samples)
                logging.info('using pca with {} holding {}'.format(len(pca.explained_variance_ratio_), np.sum(pca.explained_variance_ratio_)))
                self.pca = pca
            self.all_samples = self.pca.transform(self.all_samples) # check out self.pca, transformed samples


class NaiveBayes4Test(Bayes4Test, NaiveBayes):
    pass

class multinomial_Bayes(Bayes):
    def _compute_class_pd_prior(self):
        self.class_pd_priors = []
        for c in range(self.n_classes):
            c_samples = self.samples[c == self.raw_labels, :].astype(np.float32)
            avg_vec = np.mean(c_samples, axis=0)
            c_samples = np.transpose(c_samples)
            cov_mat = np.cov(c_samples)
            self.class_pd_priors.append((avg_vec, cov_mat))

    def _multi_gaussian_log_density(self, x_vec, c):
        avg_vec, cov_mat = self.class_pd_priors[c]
        result = multi_gauss.logpdf(x_vec, avg_vec, cov_mat, allow_singular=True)

        return result

    def _compute_class_probas(self, sample):

        # avg_vec, cov_mat = self.class_pd_priors[c]
        return [self._multi_gaussian_log_density(sample, c) for c in range(self.n_classes)]

class multi_Bayes4Test(multinomial_Bayes, Bayes4Test_PCA):
    pass
    # def _prepare_labels(self):
    #     mappings = lambda x:0 if x==0 else (1 if x<0 else 2)
    #     self.raw_labels = np.array([mappings(a[0]*a[1]-a[2]*a[3])for a in self.tmp])
    # #self.tmp is meant to keep the pre-transform data

class ups_multi_Bayes4Test(multinomial_Bayes, Bayes4Test_PCA):
    def _read_data(self, file, train=True):
        # init for self.samples and self.raw_inputs
        self.all_samples = []
        self.all_raw_labels = []
        # self.cls = []
        with open(file) as f:
            for line in f.readlines():
                fields = line.split(' ')
                # obtaining label
                label = fields[0]
                if train:
                    if label in self.cls:
                        label = self.cls.index(label)
                    else:
                        tmp = label
                        label = len(self.cls)
                        self.cls.append(tmp)
                    self.all_raw_labels.append(label)
                else:# test
                    self.all_raw_labels.append(self.cls.index(label))

                # obtaining attrs
                attrs = [eval(f.split(':')[-1]) for f in fields[1:] if f!='\n']
                self.all_samples.append(attrs)

        self.all_samples = np.array(self.all_samples, dtype=np.float64)
        self.all_raw_labels = np.array(self.all_raw_labels, dtype=np.int)
        if train:
            self.n_all_samples = len(self.all_samples)
            self.n_attribute = len(self.all_samples[0])
            self.n_classes = len(self.cls)

def threading_data(data=None, fn=None, thread_count=None, **kwargs):

    def apply_fn(results, i, data, kwargs):
        results[i] = fn(data, **kwargs)

    if thread_count is None:
        results = [None] * len(data)
        threads = []
        # for i in range(len(data)):
        #     t = threading.Thread(name='threading_and_return', target=apply_fn, args=(results, i, data[i], kwargs))
        for i, d in enumerate(data):
            t = threading.Thread(name='threading_and_return', target=apply_fn, args=(results, i, d, kwargs))
            t.start()
            threads.append(t)
    else:
        divs = np.linspace(0, len(data), thread_count + 1)
        divs = np.round(divs).astype(int)
        results = [None] * thread_count
        threads = []
        for i in range(thread_count):
            t = threading.Thread(name='threading_and_return', target=apply_fn, args=(results, i, data[divs[i]:divs[i + 1]], kwargs))
            t.start()
            threads.append(t)

    for t in threads:
        t.join()

    if thread_count is None:
        try:
            return np.asarray(results)
        except Exception:
            return results
    else:
        return np.concatenate(results)

def main():
    logger = 'loggers/logger_{}.txt'
    dataset = 'uspst'
    train_files = {'balance':(r'datasets/balance_uni_train.xls', r'datasets/balance_gnd_train.xls'),
                   'adult':'datasets/adult.data',
                   'ups':'datasets/usps',
                   'uspst':(r'datasets/uspst_uni_train.xls', r'datasets/uspst_gnd_train.xls')}
    test_files = {'balance':(r'datasets/balance_uni_test.xls', None),
                  'adult':'datasets/adult.test',
                  'ups':'datasets/usps.t',
                  'uspst':(r'datasets/uspst_uni_test.xls', None)}
    cls = {'balance':multi_Bayes4Test,
           'adult':NaiveBayes,
           'ups':ups_multi_Bayes4Test,
           'uspst':multi_Bayes4Test}

    cont_values = {'balance':None,
                   'adult': [0, 2, 4, 10, 11, 12],
                   'ups':None,
                   'uspst':None}

    trainf = train_files[dataset]
    testf = test_files[dataset]

    Method = cls[dataset]
    nb = Method(continuous_attr=cont_values[dataset], logger=logger.format(dataset), dataset=dataset,
                use_validation=False, train_ratio=0.9, use_pca=False, n_components=20)
    nb.train(file=trainf)

    #============== this is the validation phase for choosing n_components ==========
    # accs = []
    # for n in range(10,50):
    #
    #     initializers = [dict(continuous_attr=cont_values[dataset], logger=logger.format(dataset), dataset=dataset,
    #                     use_validation=True, train_ratio=0.9, use_pca=True, n_components=n)]*20
    #     th_fn = lambda d:Method(**d).train(trainf)
    #     acc = np.mean(threading_data(initializers, fn=th_fn))
    #     # for _ in range(20):
    #     #     nb = Method()
    #     #     acc += nb.train(file=trainf)
    #     # acc /= 20.
    #     accs.append(acc)
    #     print('n={},acc={}'.format(n,acc))
    #
    # import matplotlib.pyplot as plt
    # plt.style.use('ggplot')
    # plt.figure(0)
    # plt.plot(np.arange(10,50), accs, color='blue', marker='o', mec='red', mfc='none')
    # plt.xlabel('pca_components')
    # plt.ylabel('accuracy ')
    # plt.grid()
    # plt.savefig('validation2.png')
    # plt.show()
    #=============================================================================
    nb.test(file=testf, verbose=True)  # testing




if __name__ == '__main__':
    main()
