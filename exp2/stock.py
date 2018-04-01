import requests as rq
from bs4 import BeautifulSoup as bs
import numpy as np
import pandas as pd
import threading
import logging


def get_html(url):
    try:
        useragent = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:59.0) Gecko/20100101 Firefox/59.0'}  # 模拟浏览器
        rsp = rq.get(url, headers=useragent)
        rsp.raise_for_status()  # 根据状态码抛出HTTPError异常
        rsp.encoding = rsp.apparent_encoding  # 使得解码正确
        logging.info('succeed in requesting url {}'.format(url))
        return rsp.text
    except Exception as e:
        logging.error("Error occurs when requesting url {}\n" + url + repr(e))
        return None



def extract_mappings():
    stock_list_url = 'http://quote.eastmoney.com/stocklist.html'
    stock_list_rsp = get_html(stock_list_url)
    soup = bs(stock_list_rsp, 'html.parser')
    tmp = soup.find('div', attrs={'id': 'quotesearch'})
    uls = tmp.find_all('ul')

    mappings = []  # a list of 2, each dict
    for ul in uls:
        mappings.append({a.string: a.attrs['href'][:-5].split(r'/')[-1] for a in ul.find_all('a')})

    return mappings


def one_stock_detail(stock):
    name, stock_id = stock

    stock_detail_url = r'https://gupiao.baidu.com/stock/{}.html'.format(stock_id)
    try:
        detail_page_rsp = get_html(stock_detail_url)

        soup = bs(detail_page_rsp, 'html.parser')

        head = soup.find('div', class_='price')

        head_values = list(stock) + [x.string for x in head.children if x != '\n']
    except Exception as e:

        logging.error('{} with id={} meets error\n{}'.format(name, stock_id, repr(e)))
        return list(stock) + [None] * 25




    lines = ('line1', 'line2')
    for line in lines:
        linetag = soup.find('div', class_=line)
        if linetag is None:
            head_values += [None] * 22
            break
        dls = linetag.find_all('dl')
        contents = [[c.contents[0] for c in dl.children if isinstance(c, type(linetag))] for dl in dls]
        line_titles, line_values = [x for x in zip(*contents)]

        head_values.extend(line_values)

    return head_values

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

def main(n_parallel, save_path):

    mappings = extract_mappings()

    titles = ['名称', 'id', '今收', '增幅', '增比', '今开', '成交量', '最高', '涨停', '内盘', '成交额', '委比', '流通市值', '市盈率', '每股收益', '总股本',
              '昨收', '换手率', '最低', '跌停', '外盘', '振幅', '量比', '总市值', '市净率', '每股净资产', '流通股本']

    for mapping in mappings:  # sh & sz

        stocks = list(mapping.items())
        rounds = (len(stocks) - 1) // n_parallel + 1
        gen = (stocks[i * n_parallel:min(len(stocks), (i + 1) * n_parallel)] for i in range(rounds))

        # pool = ProcessPool(n_parallel)

        start = True
        for batch in gen:

            med_results = threading_data(batch, one_stock_detail)
            print('finish {}'.format(batch))

            if start:
                df = pd.DataFrame(med_results, columns=titles)
                df.to_csv(save_path, index=False)
                start = False
            else:  # https://gupiao.baidu.com/stock/sh500009.html
                df = pd.DataFrame(med_results)
                df.to_csv(save_path, mode='a', index=False, header=False)

if __name__ == '__main__':
    #========= params ==========
    n_parallel = 20
    save_path = 'results.csv'
    # logger = None则不写入文件，直接控制台显示信息
    logger = 'logger.txt'
    #==========================
    if logger is None:
        logging.basicConfig(level=logging.INFO, format='%(message)s')
    else:
        logging.basicConfig(level=logging.INFO, format='%(message)s', filemode='w', filename='logger.txt')

    main(n_parallel, save_path)
