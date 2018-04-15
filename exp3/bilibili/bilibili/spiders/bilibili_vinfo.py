# -*- coding: utf-8 -*-
import scrapy
import logging
from json import loads # for json
import time
from exp3.bilibili.bilibili.items import BilibiliItem

class BilibiliVinfoSpider(scrapy.Spider):
    name = "bilibili_vinfo"
    allowed_domains = ["bilibili.com"]
    basic_search_link = 'https://s.search.bilibili.com/cate/search?main_ver=v3&search_type=video&view_type=hot_rank&copy_right=-1'

    def start_requests(self): # starting url is the main site
        # attention! the scheme 'https' must be added
        yield scrapy.Request('https://www.bilibili.com', callback=self.extract_section_links)

    def extract_section_links(self, response):
        # from the mainsite's rsp, we extract section related infos
        navs = response.xpath('//ul[@class="nav-menu"]/li/a')
        links = navs.xpath('./@href').extract()
        sec_names = navs.xpath('./div/text()').extract()
        links = links[:len(sec_names)] # num of links is more than names
        for link,name in zip(links,sec_names):
            url = 'https:'+link
            metadata = dict(secname=name)
            yield scrapy.Request(url, meta=metadata, callback=self.extract_subsect)

    def _construct_search_link(self, n_query, link_params, current_date=None):
        if current_date is None: # haven't checked the format
            current_date = time.strftime('%Y%m%d', time.localtime())
        last_date = int_date = int(current_date)
        # check if the hundred is larger than 0 after subtraction,e.g. 20180415
        hundredth = int(current_date[-4:-2])
        for i in range(n_query):
            if hundredth>3:# at least subtracted to 01
                hundredth -= 3
                int_date -= 300
            else : #e.g. 20180215 - 3 months => 20171115
                hundredth += 9 # 12-3
                int_date += 900
                int_date -= 10000
            #=== construct the link ===
            default_params = {'order': 'stow',#conf.searches_ranked_by,
                              'cate_id': 17, # temp value
                              'pagesize': '30',
                              'page': 1,
                              'time_from': int_date,
                              'time_to': last_date}
            last_date = int_date
            default_params.update(link_params)
            url = self.basic_search_link
            for k,v in default_params.items():
                url += '&{}={}'.format(k,v)
            yield url

    def extract_subsect(self, response):
        json_str = response.xpath('//script/text()').re('window.__INITIAL_STATE__=(.*?);')
        if len(json_str) == 0:
            logging.info('no json data can be extracted from url={}'.format(response.url))
            return
        else: # not sure if there will be more than 1
            json_str = json_str[0]

        try :
            json_dict = loads(json_str)
            subsect_dictlist = json_dict['config']['sub']
        except :
            logging.info('invalid format json date extracted from url={}'.format(response.url))
            return

        for subsect in subsect_dictlist:
            metadata = response.meta # existing meta has section name
            metadata.update(dict(subsec_name=subsect['name'], subsec_id=subsect['tid']))
            for url in self._construct_search_link(4, link_params=dict(cat_id=subsect['tid'])):
                yield scrapy.Request(url, callback=self.extract_search_result, meta=metadata)

    def extract_search_result(self, response):
        metadata = response.meta
        try:
            rank_list = loads(response.text)
        except:
            logging.info('search failed for subsect={} with id={}'.format(metadata['subsec_name'], metadata['subsec_id']))
            return
        # succeeds
        rank_list = rank_list['result']
        for item_dict in rank_list:
            yield BilibiliItem(sec_name=metadata['secname'],
                               subsec_name=metadata['subsec_name'],
                               vid_name=item_dict['title'],
                               vid_id=item_dict['id'],
                               vid_author=item_dict['author'],
                               vid_pubdate=item_dict['pubdate'],
                               vid_play=item_dict['play'],
                               vid_review=item_dict['review'],
                               vid_danmu=item_dict['video_review'],
                               vid_fav=item_dict['favorites'])

if __name__ == '__main__':
    import pandas as pd

    filepath = r'..\..\results.json'
    df = pd.read_json(filepath, encoding='utf-8')
    df.to_excel(r'..\..\results.xls', header=True, index=False)




