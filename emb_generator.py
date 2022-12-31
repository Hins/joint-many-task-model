import sys
import os
import multiprocessing
import jieba
from gensim.models import Word2Vec
import json
import numpy

'''
train_sentence = []
for root, dirs, files in os.walk(sys.argv[1]):
    for file in files:
        path = os.path.join(root, file)
        with open(path, 'r') as f:
            for line in f:
                line = line.replace("\n", "").replace(" ", "")
                json_obj = json.loads(line)
                elements = json_obj.get("text").replace("<br>", "").split("\n\n")
                for element in elements:
                    for sentence in element.split("。"):
                        sentence = sentence.replace("\n", "")
                        if sentence.strip() == "" or sentence.isspace():
                            continue
                        tokens = [token for token in jieba.cut(sentence) if token.isspace() is False]
                        if len(tokens) >= 5:
                           train_sentence.append(tokens)
            print(len(train_sentence))
            print(path + " complete")

print(len(train_sentence))

with open("./dep_train.txt", "w") as f:
    for sentence in train_sentence:
        f.write("".join(sentence) + "\n")
'''
train_sentence = []
with open("/home/sjt/xtpan/graph-dependency-parsing-master/dep_train.txt", 'r') as f:
    for line in f:
        tokens = [token for token in jieba.cut(line.replace("\n", "")) if token.isspace() is False]
        if len(tokens) >= 5:
            train_sentence.append(tokens)

model = Word2Vec(train_sentence, vector_size=100, window=5, min_count=5, workers=multiprocessing.cpu_count())
model.save("/home/sjt/xtpan/word2vec.model")
#model.wv.save_word2vec_format("./cn_vectors.txt", binary=False)
