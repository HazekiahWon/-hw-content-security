# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class BilibiliItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    sec_name = scrapy.Field()
    subsec_name = scrapy.Field()
    vid_name = scrapy.Field()
    vid_id = scrapy.Field()
    vid_author = scrapy.Field()
    vid_pubdate = scrapy.Field()
    vid_play = scrapy.Field()
    vid_review = scrapy.Field()
    vid_fav = scrapy.Field()
    vid_danmu = scrapy.Field()

