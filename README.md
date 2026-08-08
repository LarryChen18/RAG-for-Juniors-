# 📚 基于RAG的智能编程助手（初级版）

这是一个基于大语言模型（Large Language Model, LLM）、视觉语言模型（Vision Language Model, VLM）以及向量数据库的检索增强生成系统（Retrieval-Augmented Generation, RAG）。

本项目实现了一个完整的多模态 RAG Pipeline，实现从非结构化文档到智能问答的全过程：

- 使用 **Qwen-VL** 完成图片OCR；
- 使用 **BGE-large-zh-v1.5** 完成文本向量化；
- 使用 **Milvus** 作为向量数据库存储知识；
- 使用 **Docker** 作为数据库容器
- 使用 **Attu** 可视化Milvus数据库的存储情况
- 使用 **Qwen3-Coder-Plus** 基于检索结果生成答案。


---

# ✨ Project Overview

传统大语言模型（LLM）存在两个主要问题：

1. 模型训练数据无法覆盖用户的私有知识；
2. 长文本直接输入模型会造成 Token 消耗增加，并降低模型对关键信息的关注能力。


因此，本项目采用 RAG 架构：

```
用户问题

    ↓

Query Embedding

    ↓

Milvus向量检索

    ↓

Top-K相关知识

    ↓

Prompt增强

    ↓

LLM生成答案
```

通过外部知识库增强大模型，使其能够基于指定文档完成更加准确、可靠的回答。



---

# 📂 Project Structure


```
RAG_Project

│
├── src
│
│   ├── OCR.py
│   │      # 文档图片OCR模块
│   │
│   ├── milvus_encode.py
│   │      # 文本切片、Embedding、Milvus建库
│   │
│   └── RAGSystem.py
│          # RAG检索与问答主程序
│
│
│
│
├── models
│      # 本地Embedding模型
│
│
├── requirements.txt
│
├── README.md
│
└── .env

```


---

# ⚙️ Environment Setup


## Requirements

测试环境：

- Windows 11
- Python 3.x
- Docker Desktop
- Milvus 2.4.12
- Attu


安装依赖：

```bash
pip install -r requirements.txt
```


主要Python依赖：

```
openai
pymilvus
sentence-transformers
python-dotenv
numpy
tqdm
```


---

# 🚀 Run The Project
## Step 0. 下载docker desktop等

下载docker网址：
````
https://www.docker.com/products/docker-desktop/
````
下载完成后安装 Docker Desktop。


安装过程中保持默认配置即可。

---

下载Attu网址：
```
https://github.com/zilliztech/attu/releases
```
## Step 1. Deploy Milvus


本项目使用 Docker 部署 Milvus 向量数据库。


下载 Milvus Docker Compose 配置（在本项目中）：


```
milvus-standalone-docker-compose.yml
```
放入项目文件夹
```
RAG

│
├── milvus
│
│   └── milvus-standalone-docker-compose.yml
│
├── src
│
├── models
│
│
└── README.md
```

启动：

先打开docker，保证还没有创建任何Milvus数据库的container

然后运行（powershell）：

```bash
docker compose -f milvus-standalone-docker-compose.yml up -d
```


检查运行状态：

```bash
docker ps
```


正常情况下：

```
milvus-standalone
milvus-etcd
milvus-minio
```

均处于运行状态。


---

# Step 2. Start Attu Visualization


Attu 是 Milvus 的可视化管理工具，用于查看：

- Database
- Collection
- Vector
- Metadata


启动（直接在Attu中连接即可），下面是命令行写法：

```bash
docker run -d \
-p 8000:3000 \
--name attu \
zilliz/attu
```


浏览器访问：

```
http://localhost:8000
```


连接：

```
localhost:19530
```


成功连接后即可查看 Milvus 中的数据。


---

# Step 3. Configure API Key


项目中使用阿里云 DashScope API 调用：

- Qwen-VL
- Qwen3-Coder-Plus

然后分别写入src中的OCR和RAG文件中的“your api key”部分

---

# Step 4. Document OCR


## Purpose

由于大量知识来源于图片等非结构化数据，需要首先进行文本提取。


OCR流程：

```
Image

↓

Base64 Encoding

↓

Qwen-VL

↓

Text Extraction

↓

OCR Result
```


运行：

```bash
python OCR.py
```


输出：

```
ocr_result.txt
```


相比传统 OCR：

视觉语言模型能够理解：

- 页面结构；
- 标题层级；
- 表格信息；
- 复杂排版。


---

# Step 5. Build Vector Database


运行：

```bash
python encoding and Milvus.py
```


该模块完成完整知识库构建流程：


## 1. Text Cleaning


清理：

- 多余空格；
- 无意义换行；
- 空文本。


目的：

提高知识库信息密度。


---

## 2. Chunk Splitting


长文本不能直接存入向量数据库。

因此：

```
Document

↓

Chunk 1

Chunk 2

Chunk 3

...
```


Chunk作为后续检索的基本单位。


---

## 3. Text Embedding


使用：

```
BAAI/bge-large-zh-v1.5
```


将文本转换为向量：

```
Text

↓

[0.23,0.56,...]
```


Embedding使计算机能够通过数学方式比较文本语义相似度。


---

## 4. Store Into Milvus


Milvus中保存：

```
id

text

vector

source
```


向量检索采用：

```
COSINE Similarity
```


通过计算用户问题向量与知识库向量之间的相似度，找到相关文本。


---

# Step 6. Run RAG System


运行：

```bash
python RAG.py
```


输入问题：

例如：

```
什么是Shell脚本？
```


系统执行：

```
User Question

↓

Embedding

↓

Milvus Search

↓

Relevant Context

↓

Prompt Construction

↓

Qwen3-Coder-Plus

↓

Final Answer
```


---

# 🔍 Core Modules


# 1. OCR Module


作用：

将非结构化图片数据转换为文本。


输入：

```
Image
```


输出：

```
Structured Text
```


使用：

```
Qwen-VL
```


优势：

相比传统OCR，可以理解复杂文档结构。


---

# 2. Retrieval Module


用户问题首先经过Embedding：

```
Question

↓

Vector
```


然后在Milvus中搜索：

```
Vector Database

↓

Top-K Similar Chunks
```


返回：

- 相关文本；
- 来源信息；
- 相似度结果。


---

# 3. Generation Module


检索结果不会直接作为答案。


而是：

```
Retrieved Context

+

User Question

↓

Prompt

↓

LLM

↓

Answer
```


LLM负责：

- 理解用户问题；
- 综合检索内容；
- 生成自然语言回答。


---

# 🐛 Engineering Problems Solved


## Problem 1: Milvus Container Conflict


错误：

```
container name "/milvus-etcd" already exists
```


原因：

Docker中存在旧Milvus容器。


解决：

删除旧容器：

```bash
docker rm -f milvus-etcd
```


重新启动Milvus。


---

## Problem 2: OCR Duplicate Recognition


问题：

同一图片被OCR多次。


原因：

图片扫描过程中重复匹配不同大小写扩展名。


例如：

```
page1.jpg

page1.JPG
```


导致同一页面进入OCR队列。


解决：

统一文件后缀：

```python
file.lower()
```


并增加：

```python
image_files.sort()
```


保证：

- 不重复；
- 页面顺序正确。


---

# 📈 Future Improvements


## 1. Add Reranker


当前流程：

```
Embedding Retrieval

↓

LLM
```


未来优化：

```
Embedding Retrieval

↓

Reranker

↓

LLM
```


进一步提升检索准确率。


---

## 2. Add Citation


当前回答：

```
Answer
```


未来增加：

```
Answer

+

Source Document

+

Page Number

+

Reference Chunk
```


提高可信度。


---

## 3. Agent Extension


当前系统：

```
RAG Question Answering
```


未来扩展：

```
RAG

↓

Agent System
```


增加：

- Tool Calling
- Memory
- Planning
- Multi-step Reasoning


---

# 📝 Personal Learning Summary


通过该项目，我完成了从理论学习到工程实践的完整 RAG 复现。


主要掌握：

- 理解 RAG 的整体架构；
- 理解 Chunk、Embedding、Vector Search 的作用；
- 掌握 Milvus 向量数据库部署；
- 完成 OCR → Embedding → Retrieval → Generation 全流程；
- 学习 Docker 环境部署；
- 独立定位并解决工程问题。


该项目作为进一步学习：

- LLM Agent；
- Tool Calling；
- Intelligent System；

的重要基础工程实践。




