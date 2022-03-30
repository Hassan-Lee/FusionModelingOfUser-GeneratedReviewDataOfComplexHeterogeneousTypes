# coding: utf-8
import numpy as np
import gensim
import pickle
import torch
from tokenization import *
from torch.utils.data import Dataset
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, Word2Vec
import sys

sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf8', buffering=1)


class DocDataset(Dataset):
    def __init__(self, taskname, txtPath=None, lang="zh", tokenizer=None, stopwords=None, no_below=5,
                 no_above=0.01, rebuild=False, use_tfidf=False, had_w2v=False, vector_size=100):
        cwd = os.getcwd()
        txtPath = os.path.join(cwd, 'data', f'{taskname}_lines.txt') if txtPath is None else txtPath
        tmpDir = os.path.join(cwd, 'data', taskname)
        self.txtLines = [line.strip('\n') for line in open(txtPath, 'r', encoding='utf-8')]
        self.dictionary = None
        self.bows, self.docs = None, None
        self.use_tfidf = use_tfidf
        self.tfidf, self.tfidf_model = None, None
        self.had_w2v = had_w2v
        self.word2vec = None
        self.vector_size = vector_size

        if not os.path.exists(tmpDir):
            os.mkdir(tmpDir)
        if not rebuild and os.path.exists(os.path.join(tmpDir, 'corpus.mm')):
            self.bows = gensim.corpora.MmCorpus(os.path.join(tmpDir, 'corpus.mm'))
            if self.use_tfidf:
                self.tfidf = gensim.corpora.MmCorpus(os.path.join(tmpDir, 'tfidf.mm'))

            self.dictionary = Dictionary.load_from_text(os.path.join(tmpDir, 'dict.txt'))
            self.docs = pickle.load(open(os.path.join(tmpDir, 'docs.pkl'), 'rb'))
            self.dictionary.id2token = {v: k for k, v in
                                        self.dictionary.token2id.items()}
        else:
            if stopwords is None:
                stopwords = set([i.strip('\n').strip() for i in
                                 open(os.path.join(cwd, 'data', 'stopwords.txt'), 'r', encoding='utf-8')])
            # self.txtLines is the list of string, without any preprocessing.
            # self.texts is the list of list of tokens.
            print('Tokenizing ...')
            if tokenizer is None:
                tokenizer = globals()[LANG_CLS[lang]](stopwords=stopwords)
            self.docs = tokenizer.tokenize(self.txtLines)
            self.docs = [line for line in self.docs if line != []]

            print('Building dictionary... ')
            self.dictionary = Dictionary(self.docs)
            print('Removing un-relevant tokens')
            self.dictionary.filter_extremes(no_below=no_below, no_above=no_above,
                                            keep_n=None)
            self.dictionary.compactify()
            print('Building id2token ...')
            self.dictionary.id2token = {v: k for k, v in
                                        self.dictionary.token2id.items()}
            # convert to BOW representation
            print('Building Bow ...')
            self.bows, _docs = [], []
            for doc in self.docs:
                _bow = self.dictionary.doc2bow(doc)
                if _bow != []:
                    _docs.append(list(doc))
                    self.bows.append(_bow)

            self.docs = _docs

            if self.use_tfidf is True:
                self.tfidf_model = TfidfModel(self.bows)
                self.tfidf = [self.tfidf_model[bow] for bow in self.bows]

            print('Serializing the dictionary ...')
            gensim.corpora.MmCorpus.serialize(os.path.join(tmpDir, 'corpus.mm'), self.bows)
            self.dictionary.save_as_text(os.path.join(tmpDir, 'dict.txt'))
            pickle.dump(self.docs, open(os.path.join(tmpDir, 'docs.pkl'), 'wb'))
            if self.use_tfidf:
                gensim.corpora.MmCorpus.serialize(os.path.join(tmpDir, 'tfidf.mm'), self.tfidf)

        self.vocabsize = len(self.dictionary)
        self.numDocs = len(self.bows)
        print(f'Processed {len(self.bows)} documents.')

        self.get_w2v_model()

    def get_w2v_model(self, window=5, min_count=5, workers=12, sg=0):

        if self.had_w2v is False:
            print('Training Word2Vec...')
            new_docs = self.docs
            wv_model = Word2Vec(sentences=new_docs,
                                vector_size=self.vector_size,
                                window=window,
                                min_count=min_count,
                                workers=workers,
                                sg=sg)
            print('Word2Vec Model Training Finished')
            self.word2vec = wv_model
        else:
            self.word2vec = Word2Vec.load('{}.model'.format(taskname))

    def save_w2v_model(self):
        if self.word2vec is None:
            print('Not get a Word2Vec Model yet,please train a model first')
            answer = input('Do you want to train a Word2Vec Model now? Please Enter yes or no')
            if answer == 'yes' or answer == 'YES' or answer == 'Yes':
                self.get_w2v_model()
            elif answer == 'no' or answer == 'NO' or answer == 'No':
                print("Can't save it ")
            else:
                print('Confirm answer you entered does not match format')
        else:
            self.word2vec.save('{}.model'.format(taskname))

    def get_wv_vector(self, same_size=True):
        if same_size == False:
            return self.word2vec.wv.vector
        else:
            w2v = []
            for key in self.dictionary.token2id.keys():
                w2v.append(self.word2vec.wv[key])
            w2v = np.array(w2v)

            return w2v

    def get_wv_input(self):
        max_sentnce_length = len(max(self.docs, key=len))
        w2v_three_dims = np.zeros((len(self.bows), max_sentnce_length, self.vector_size))
        for i, doc in enumerate(self.docs):
            for j, key in enumerate(doc):
                if self.word2vec.wv.__contains__(key):
                    w2v_three_dims[i][j] = self.word2vec.wv[key]
        return w2v_three_dims

    def __getitem__(self, idx):
        bow = torch.zeros(self.vocabsize)
        if self.use_tfidf:
            item = list(zip(*self.tfidf[idx]))
        else:
            item = list(zip(*self.bows[idx]))

        # bow = [[token_id1,token_id2,...],[freq1,freq2,...]]
        bow[list(item[0])] = torch.tensor(list(item[1])).float()
        txt = self.docs[idx]
        wv_input = self.get_wv_input()
        wv = wv_input[idx]
        return txt, bow, wv

    def __len__(self):
        return self.numDocs

    def collate_fn(self, batch_data):
        texts, bows = list(zip(*batch_data))
        return texts, torch.stack(bows, dim=0)

    def __iter__(self):
        for doc in self.docs:
            yield doc

    def show_dfs_topk(self, topk=20):
        ndoc = len(self.docs)
        dfs_topk = sorted([(self.dictionary.id2token[k], fq) for k, fq in self.dictionary.dfs.items()],
                          key=lambda x: x[1], reverse=True)[:topk]
        for i, (word, freq) in enumerate(dfs_topk):
            print(f'{i + 1}:{word} --> {freq}/{ndoc} = {(1.0 * freq / ndoc):>.13f}')
        return dfs_topk

    def show_cfs_topk(self, topk=20):
        ntokens = sum([v for k, v in self.dictionary.cfs.items()])
        cfs_topk = sorted([(self.dictionary.id2token[k], fq) for k, fq in self.dictionary.cfs.items()],
                          key=lambda x: x[1], reverse=True)[:topk]
        for i, (word, freq) in enumerate(cfs_topk):
            print(f'{i + 1}:{word} --> {freq}/{ntokens} = {(1.0 * freq / ntokens):>.13f}')

    def topk_dfs(self, topk=20):
        ndoc = len(self.docs)
        dfs_topk = self.show_dfs_topk(topk=topk)
        return 1.0 * dfs_topk[-1][-1] / ndoc


class TestData(Dataset):
    def __init__(self, dictionary=None, txtPath=None, lang="zh", vector_size=100, tokenizer=None, stopwords=None,
                 use_tfidf=False, had_w2v=False):
        cwd = os.getcwd()
        self.txtLines = [line.strip('\n') for line in open(txtPath, 'r', encoding='utf-8')]
        self.dictionary = dictionary
        self.bows, self.docs = None, None
        self.use_tfidf = use_tfidf
        self.tfidf, self.tfidf_model = None, None
        self.had_w2v = had_w2v
        self.word2vec = None
        self.vector_size = vector_size
        self.wv = None

        if stopwords is None:
            stopwords = set([j.strip('\n').strip() for j in
                             open(os.path.join(cwd, 'data', 'stopwords.txt'), 'r', encoding='utf-8')])

        print('Tokenizing ...')
        if tokenizer is None:
            tokenizer = globals()[LANG_CLS[lang]](stopwords=stopwords)
        self.docs = tokenizer.tokenize(self.txtLines)
        # convert to BOW representation
        self.bows, _docs = [], []
        for doc in self.docs:
            if doc is not None:
                _bow = self.dictionary.doc2bow(doc)
                if _bow != []:
                    _docs.append(list(doc))
                    self.bows.append(_bow)
                else:
                    _docs.append(None)
                    self.bows.append(None)
            else:
                _docs.append(None)
                self.bows.append(None)
        self.docs = _docs
        if self.use_tfidf:
            self.tfidf_model = TfidfModel(self.bows)
            self.tfidf = [self.tfidf_model[bow] for bow in self.bows]
        self.vocabsize = len(self.dictionary)
        self.numDocs = len(self.bows)
        print(f'Processed {len(self.bows)} documents.')

    def get_w2v_model(self, window=5, min_count=5, workers=12, sg=0):

        if self.had_w2v is False:
            print('Training Word2Vec...')
            new_docs = self.docs
            wv_model = Word2Vec(sentences=new_docs,
                                vector_size=self.vector_size,
                                window=window,
                                min_count=min_count,
                                workers=workers,
                                sg=sg)
            print('Word2Vec Model Training Finished')
            self.word2vec = wv_model
        else:
            self.word2vec = Word2Vec.load('{}.model'.format(taskname))

    def save_word2vec(self):
        if self.word2vec is None:
            print('Not get a Word2Vec Model yet,please train a model first')
            answer = input('Do you want to train a Word2Vec Model now? Please Enter yes or no')
            if answer == 'yes' or answer == 'YES' or answer == 'Yes':
                self.get_w2v_model()
            elif answer == 'no' or answer == 'NO' or answer == 'No':
                print("Can't save it ")
            else:
                print('Confirm answer you entered does not match format')
        else:
            self.word2vec.save('{}.model'.format(taskname))

    def get_wv_vector(self, same_size=True):
        if not same_size:
            return self.word2vec.wv.vector
        else:
            w2v = []
            for key in self.dictionary.token2id.keys():
                w2v.append(self.word2vec.wv[key])
            w2v = np.array(w2v)
            return w2v

    def __getitem__(self, idx):
        bow = torch.zeros(self.vocabsize)
        if self.use_tfidf:
            item = list(zip(*self.tfidf[idx]))
        else:
            item = list(zip(*self.bows[idx]))  # bow = [[token_id1,token_id2,...],[freq1,freq2,...]]
        bow[list(item[0])] = torch.tensor(list(item[1])).float()
        txt = self.docs[idx]
        return txt, bow

    def __len__(self):
        return self.numDocs

    def __iter__(self):
        for doc in self.docs:
            yield doc


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    taskname = 'software_5'
    lang = 'en'
    no_below = 5
    no_above = 0.005
    rebuild = False
    docSet = DocDataset(taskname,
                        lang=lang,
                        no_below=no_below,
                        no_above=no_above,
                        rebuild=rebuild,
                        use_tfidf=False,
                        vector_size=150)

    print('docSet.vocabsize:', docSet.vocabsize)
    dataloder = DataLoader(docSet, batch_size=125)

    for iter, data in enumerate(dataloder):
        txt, bow, wv = data
        print(wv.shape)
        break

    # 词云绘图
    # doclist = []
    # for key in docSet.dictionary.iterkeys():
    #    doclist.append([docSet.dictionary.get(key), docSet.dictionary.dfs[key]])

    # pd.DataFrame(doclist).to_csv('software_wordcloud_data.csv')
    # print('Finished')
