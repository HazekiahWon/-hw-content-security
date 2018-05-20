#!/usr/bin/env python

'''
SVM and KNearest digit recognition.

Sample loads a dataset of handwritten digits from '../data/digits.png'.
Then it trains a SVM and KNearest classifiers on it and evaluates
their accuracy.

Following preprocessing is applied to the dataset:
 - Moment-based image deskew (see deskew())
 - Digit images are split into 4 10x10 cells and 16-bin
   histogram of oriented gradients is computed for each
   cell
 - Transform histograms to space with Hellinger metric (see [1] (RootSIFT))


[1] R. Arandjelovic, A. Zisserman
    "Three things everyone should know to improve object retrieval"
    http://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf

Usage:
   digits.py
'''


# Python 2/3 compatibility
from __future__ import print_function

# built-in modules
from multiprocessing.pool import ThreadPool

import cv2 as cv

import numpy as np
from numpy.linalg import norm

# local modules
from common import clock, mosaic



SZ = 16 # size of each digit is SZ x SZ
BIN_LEN = SZ // 2
CLASS_N = 10
DIGITS_FN = '../data/digits.png'

def split2d(img, cell_size, flatten=True):
    h, w = img.shape[:2]
    sx, sy = cell_size
    cells = [np.hsplit(row, w//sx) for row in np.vsplit(img, h//sy)]
    cells = np.array(cells)
    if flatten:
        cells = cells.reshape(-1, sy, sx)
    return cells

def load_digits(usps_dataset, gamma=0.5):
    file = usps_dataset
    digits = []
    labels = []
    with open(file) as f:
        for line in f.readlines():
            fields = line.split(' ')
            # obtaining label
            labels.append(eval(fields[0])-1)
            digits.append(np.array([eval(f.split(':')[-1]) for f in fields[1:] if f != '\n']).reshape((16,16)))
        digits = np.stack(digits)
        labels = np.array(labels)

    rg = digits.max()-digits.min()
    digits = (digits-digits.min()) / rg
    digits = np.power(digits, gamma)

    # print('loading "%s" ...' % fn)
    # digits_img = cv.imread(fn, 0)
    # digits = split2d(digits_img, (SZ, SZ))
    # labels = np.repeat(np.arange(CLASS_N), len(digits)/CLASS_N)
    #===================================================
    # from sklearn.datasets import load_digits
    # mnist = load_digits()
    # digits,labels = mnist['images'],mnist['target']
    #===================================================
    return digits, labels


def deskew(img):
    # mu-中心矩，m-几何矩
    # mu(p,q) = \sigma(x,y) x^p*y^q*f(x,y) 轮廓中所有像素灰度按位置的某种加权和
    # m(p,q) = \sigma(x,y) (x-\ba{x})^p*(y-\ba{y})^q*f(x,y)
    # \ba{x} = m(1,0)/m(0,0), \ba{y} = m(0,1)/m(0,0)
    # note m(0,0) is the luma intensity sum in the area

    m = cv.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv.warpAffine(img, M, (SZ, SZ), flags=cv.WARP_INVERSE_MAP | cv.INTER_LINEAR)
    return img

class StatModel(object):
    def load(self, fn):
        self.model.load(fn)  # Known bug: https://github.com/opencv/opencv/issues/4969
    def save(self, fn):
        self.model.save(fn)

class KNearest(StatModel):
    def __init__(self, k = 3):
        self.k = k
        self.model = cv.ml.KNearest_create()

    def train(self, samples, responses):
        self.model.train(samples, cv.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        _retval, results, _neigh_resp, _dists = self.model.findNearest(samples, self.k) # give batch of samples
        return results.ravel()

class SVM(StatModel):
    def __init__(self, C = 1, gamma = 0.5):
        self.model = cv.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv.ml.SVM_RBF)
        self.model.setType(cv.ml.SVM_C_SVC)

    def train(self, samples, responses):
        self.model.train(samples, cv.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        return self.model.predict(samples)[1].ravel()


def evaluate_model(model, digits, samples, labels):
    resp = model.predict(samples)
    err = (labels != resp).mean()
    print('error: %.2f %%' % (err*100))
    # confusion table, where x is ground-truth and y is predicted
    confusion = np.zeros((10, 10), np.int32)
    for i, j in zip(labels, resp):
        confusion[i, int(j)] += 1
    print('confusion matrix:')
    print(confusion)
    print()
    # if not correctly predicted, set 0 all channels except Red to make it red
    vis = []
    for img, flag in zip(digits, resp == labels):
        img = np.uint8((img-img.min()) / (img.max()-img.min()) * 255.)
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        if not flag:
            img[...,:2] = 0
        vis.append(img)
    return mosaic(25, vis)

def preprocess_simple(digits):
    return np.float32(digits).reshape(-1, SZ*SZ) / 255.0

def preprocess_hog(digits):
    samples = []
    for img in digits:
        # image grads
        gx = cv.Sobel(img, -1, 1, 0)  #sobel算子 边缘检测 一阶差分滤波器
        gy = cv.Sobel(img, -1, 0, 1)
        # norm and angles
        mag, ang = cv.cartToPolar(gx, gy)  #极坐标变换 （模 角度）
        bin_n = 16  #区间数
        # normalize to 16 stepcases
        bin = np.int32(bin_n*ang/(2*np.pi))
        bin_cells = bin[:BIN_LEN,:BIN_LEN], bin[BIN_LEN:,:BIN_LEN], bin[:BIN_LEN, BIN_LEN:], bin[BIN_LEN:,BIN_LEN:]
        # norm as weights for each cell
        mag_cells = mag[:BIN_LEN,:BIN_LEN], mag[BIN_LEN:,:BIN_LEN], mag[:BIN_LEN, BIN_LEN:], mag[BIN_LEN:,BIN_LEN:]
        # 16 bins for each cell, a weighted count for each stepcase
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]    #统计梯度直方图
        hist = np.hstack(hists)


        # transform to Hellinger kernel  to quantify the similarity of two probability distributions
        eps = 1e-7
        hist /= hist.sum() + eps # normalize by summing to 1
        hist = np.sqrt(hist)
        # hist /= norm(hist) + eps # ??? sum(sqrt(x)^2)

        samples.append(hist)
    return np.float32(samples)

def show(name,im):
    cv.imshow(name,im)
    cv.waitKey(0)
    cv.destroyWindow(name)


if __name__ == '__main__':
    # print(__doc__)

    digits, labels = load_digits(r'..\data\usps') #图像切分 导入

    print('preprocessing...')
    # shuffle digits
    rand = np.random.RandomState(321)   #随机种子321
    shuffle = rand.permutation(len(digits))
    digits, labels = digits[shuffle], labels[shuffle]   #打乱数字顺序
    # deskew
    deskewed_digits = list(map(deskew, digits)) #纠正图片倾斜
    # hog
    samples = preprocess_hog(deskewed_digits)   #计算hog特征
    # split into validation sets
    train_n = int(0.8*len(samples)) #划分训练测试集
    print(train_n)
    # what is mosaic?

    show('test set', mosaic(25, digits[train_n:]))

    # the original digits
    digits_train, digits_test = np.split(deskewed_digits, [train_n])
    # the descriptors
    samples_train, samples_test = np.split(samples, [train_n])
    labels_train, labels_test = np.split(labels, [train_n])


    print('training KNearest...')   #knn分类器
    model = KNearest(k=4)
    model.train(samples_train, labels_train) # provide x, y
    vis = evaluate_model(model, digits_test, samples_test, labels_test)
    show('KNearest test', vis)

    print('training SVM...')    #SVM分类器
    model = SVM(C=2.67, gamma=5.383)
    model.train(samples_train, labels_train)
    vis = evaluate_model(model, digits_test, samples_test, labels_test)
    show('SVM test', vis)
    print('saving SVM as "digits_svm.dat"...')
    model.save('digits_svm.dat')

    cv.waitKey(0)
