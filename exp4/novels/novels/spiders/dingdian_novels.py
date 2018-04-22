# -*- coding: utf-8 -*-
import scrapy


class DingdianNovelsSpider(scrapy.Spider):
    name = "dingdian_novels"
    allowed_domains = ["http://www.booktxt.net/xiaoshuodaquan/"]
    start_urls = ['http://http://www.booktxt.net/xiaoshuodaquan//']

    def parse(self, response):
        pass
