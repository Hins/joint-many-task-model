# 基于JMT的中文语块分析推理工具

## 目录

+ <a href="#1">功能介绍</a>
+ <a href="#2">上手指南</a>
  + <a href="#3">开发前的配置要求</a>
  + <a href="#4">安装步骤</a>
+ <a href="#5">文件目录说明</a>

## <span name="1">功能介绍</span>

​		基于JMT的中文语块分析推理工具，预测中文语块分析结果。输入的格式为 .txt 输出格式为 .json。

##<span name="2">上手指南 </span>

### <span name="3">开发前的配置要求</span>

arm服务器
numpy==1.16.2
gensim==3.2
psutil
pathlib
jieba

### <span name="4">安装步骤</span>

pip install -r requirements.txt

## <span name="5">文件目录说明</span>

code
├── README.md ---> 工具说明
├── Dockerfile ---> docker镜像工具
├── /data/ ---> 数据文件
├── helper.py ---> util工具
├── jmt.py ---> 模型工具
├── my_lstm.py ---> lstm层工具
├── train.py ---> 训练工具
├── inference.py ---> 推理工具
├── monitoring.py ---> 监控工具
│── requirements.txt ---> 环境安装包信息
