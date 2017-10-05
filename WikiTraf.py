import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
import re

train = pd.read_csv('./train_1.csv').fillna(0)

def get_language(page):
    res = re.search('[a-z][a-z].wikipedia.org',page)
    if res:
        return res.group(0)
    return 'na'

train['lang'] = train.Page.map(get_language)
#train['text']
from collections import Counter
print(Counter(train.lang))

def get_text(page):
    res = re.search('[a-z][a-z].wikipedia.org',page)
    if res:
        foo=res.group(0)
        #print(foo[:2])
        bar = " ".join(page[:page.index(foo)].split("_"))
        #print(bar)
        #tra = translator.translate(bar)
        #print(tra.text)
        #return tra.text
        return bar
    return 'na'

train['text'] = train.Page.map(get_text)


from googletrans import Translator
translator = Translator()


