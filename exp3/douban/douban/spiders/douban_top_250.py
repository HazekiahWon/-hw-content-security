# -*- coding: utf-8 -*-
import scrapy
from ..items import DoubanItem


class DoubanTop250Spider(scrapy.Spider):
    name = "douban_top_250"

    allowed_domains = ["douban.com"]
    # start_urls = ['http://douban.com/']

    headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:59.0) Gecko/20100101 Firefox/59.0'}

    def start_requests(self):
        for i in range(0, 250, 25):
            url = 'https://movie.douban.com/top250?start={}&filter='.format(i)
            yield scrapy.Request(url, headers=self.headers)


    def parse(self, response):
        for item in response.xpath('//div[@class="info"]'):
            title = ''.join(item.xpath('.//a//span[position()<3]/text()').extract())
            directors = ' '.join([x.strip('\n').strip(' ') for x in item.xpath('.//div[@class="bd"]//p[1]//text()').extract()])
            # directors = directors.strip('\n').strip(' ')
            score, n_raters = item.xpath('.//div[@class="star"]//span//text()').extract()
            yield DoubanItem(name=title, director=directors, score=score, n_raters=n_raters)

