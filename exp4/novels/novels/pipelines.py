# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html

import os
import logging
class NovelsPipeline(object):
    save_folder = 'save'
    def __init__(self):
        if not os.path.exists(self.save_folder):
            os.mkdir(self.save_folder)

    def process_item(self, item, spider):
        book = item['bookname']
        chap = item['chapname']
        content = item['chapter']
        novelpath = os.path.join(self.save_folder, book+'.txt')
        mode = 'a' if os.path.exists(novelpath) else 'w'
        with open(novelpath, mode=mode, encoding='utf-8') as f:
            f.write(chap)
            f.write(content)

            # logging.info('written')

        return item
