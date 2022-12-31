from jmt import JMT
import tensorflow as tf
import pprint
from pathlib import Path
import logging
import os
import jieba
import time
import json

pp = pprint.PrettyPrinter(indent=4)

def main(input_path, result_path):
    model = JMT(100, 1e-5, 0.001)
    model.load_data()
    json_obj = []
    with tf.Graph().as_default() as graph:
        model.build_model()

        for file in Path(input_path).glob('**/*.txt'):
            logging.info("file{}".format(file))
            mid = os.path.relpath(str(file), input_path)
            logging.info("mid{}".format(mid))
            dst_json = os.path.join(result_path, os.path.dirname(mid), str(file.stem) + '.json')
            logging.info("dst_json{}".format(dst_json))
            os.makedirs(os.path.dirname(dst_json), exist_ok=True)

            sentences = []
            with open(str(file), "r") as f:
                for line in f:
                    sentences.append(" ".join([word for word in jieba.cut(line.replace("\n", ""))]))

            task_desc = {
                'chunk': sentences
            }
            with tf.compat.v1.Session(graph=graph) as sess:
                model.saver.restore(sess, tf.compat.v1.train.latest_checkpoint("./saves/"))
                test_start_time = time.time()
                res = model.get_predictions(sess, graph, task_desc)
                test_time = time.time() - test_start_time

                for idx, l in enumerate(res):
                    dict = {}
                    dict["id"] = idx
                    tokens = sentences[idx].split()
                    dict["text"] = ("").join(tokens)
                    dict["words"] = {}
                    for id, label in enumerate(l):
                        dict["words"][tokens[id]] = label
                    json_obj.append(dict)
                with open(dst_json, 'w', encoding='utf-8') as f:
                    json.dump(json_obj, f, indent=4, ensure_ascii=False)
            return len(json_obj), test_time, json_obj
