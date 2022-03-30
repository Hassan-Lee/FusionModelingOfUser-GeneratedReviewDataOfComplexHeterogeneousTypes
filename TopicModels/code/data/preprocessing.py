# encoding:utf-8
# @version:3.9
# @file     :preprocessing.py
# @Time     :2022 03 2022/3/4 16:26
# @Author   :ZiyuFan
import pandas as pd
import gzip
import json


def parse(path):
    g = gzip.open(path, 'rb')
    for j in g:
        yield json.loads(j)


def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


df = getDF('software_5.json.gz')['reviewText']
df.to_csv('software_5_lines.txt')
