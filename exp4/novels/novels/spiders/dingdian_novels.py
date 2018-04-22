# -*- coding: utf-8 -*-
import scrapy
from ..items import NovelsItem
import logging


class DingdianNovelsSpider(scrapy.Spider):
    name = "dingdian_novels"
    # allowed_domains = ["http://www.booktxt.net/xiaoshuodaquan/"]
    start_urls = ['http://www.booktxt.net/xiaoshuodaquan/']

    def parse(self, response):
        sections = response.xpath('//div[@class="novellist"]')
        # logging.info(sections)
        for section in sections[1:2]:
            books = section.xpath('.//li/a')
            for book in books:
                url= book.xpath('./@href').extract_first()
                name = book.xpath('./text()').extract_first()
                metadata = dict(book_name=name)
                yield scrapy.Request(url, meta=metadata, callback=self.parse_book)

    def parse_book(self, response):
        dds = response.xpath('//dt[2]/following-sibling::dd/a')
        for chapter in dds:
            cname = chapter.xpath('./text()').extract_first()
            curl = response.url+chapter.xpath('./@href').extract_first().split('/')[-1]
            metadata = response.meta
            metadata.update(dict(chapter_name=cname))
            yield scrapy.Request(curl, meta=metadata, callback=self.parse_chapter)

    def parse_chapter(self, response):
        text = response.xpath('//div[@id="content"]/text()').extract()
        texts = [x.strip('\r\n') for x in text if x !='\r\n']
        chapter = '\n'.join(texts)
        metadata = response.meta

        yield NovelsItem(bookname=metadata['book_name'], chapname=metadata['chapter_name'], chapter=chapter)

