import os
import re#正则表达式库，用于文本清洗
from sentence_transformers import SentenceTransformer#包含embedding用于向量化
from pymilvus import MilvusClient#连接向量数据库
import numpy as np

# 设置模型路径
model_path = r"your path\models\bge-large-zh-v1.5"#r""代表原始字符串，告诉python不要将其视作特殊字符
ocr_file_path = r"your path\ocr_result.txt"

# 加载本地BGE模型
print("正在加载BGE模型...")
model = SentenceTransformer(model_path)
print("模型加载完成！")

# 读取OCR结果文件
print("正在读取OCR结果文件...")
with open(ocr_file_path, 'r', encoding='utf-8') as f:#open打开文件，加上with则会将下面的操作执行后自动关闭文件
    content = f.read()

# 文本预处理和分段
def preprocess_text(text):
    """预处理文本，去除多余空格和换行"""
    # 去除多余的空格和换行
    text = re.sub(r'\s+', ' ', text)#‘\s'代表所有的空白字符（空格，换行），然后把它们都替换为一个空格
    text = text.strip()#strip（）：移除字符串两端的所有空白字符
    return text

def split_text_into_chunks(text, max_length=200):
    """将长文本分割成较小的块"""
    # 按句号分割
    sentences = re.split(r'[。！？]', text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # 如果当前块加上新句子不超过最大长度，则添加
        if len(current_chunk + sentence) <= max_length:
            current_chunk += sentence + "。"
        else:
            # 如果当前块不为空，保存它
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + "。"
    
    # 添加最后一个块
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

# 预处理文本
processed_content = preprocess_text(content)
print(f"原始文本长度: {len(content)}")
print(f"处理后文本长度: {len(processed_content)}")

# 分割文本
text_chunks = split_text_into_chunks(processed_content)
print(f"文本分割成 {len(text_chunks)} 个块")

# 对每个文本块进行编码
print("正在对文本进行向量编码...")
embeddings = model.encode(text_chunks)#每个文本对应一个向量，embeddings中存储每个文本对应的向量
print(f"编码完成！向量维度: {embeddings.shape}")

# 连接到Milvus数据库
print("正在连接Milvus数据库...")
client = MilvusClient(uri="http://localhost:19530")

# 创建或使用数据库
database_name = "your name"
try:
    client.create_database(database_name)
    print(f"创建数据库: {database_name}")
except Exception as e:
    print(f"数据库可能已存在: {e}")

client.use_database(database_name)

# 创建集合（注意：BGE-large-zh-v1.5的向量维度是1024）
collection_name = "ocr_text_collection"
vector_dimension = embeddings.shape[1]  # BGE模型的向量维度

print(f"创建集合，向量维度: {vector_dimension}")
try:
    client.create_collection(
        collection_name=collection_name,
        dimension=vector_dimension,
        metric_type="COSINE",  # 使用余弦相似度
        index_type="IVF_FLAT",
        params={"nlist": 1024}
    )
    print(f"集合 {collection_name} 创建成功")
except Exception as e:
    print(f"集合可能已存在: {e}")

# 准备往创建的集合中插入数据，每条数据中包含id，原文文本，vector，来源
insert_data = []
for i, (chunk, embedding) in enumerate(zip(text_chunks, embeddings)):
    insert_data.append({
        "id": i,
        "text": chunk,
        "vector": embedding.tolist(),
        "source": "OCR_result.txt"
    })

# 插入数据到Milvus
print("正在插入数据到Milvus...")
try:
    result = client.insert(
        collection_name=collection_name,
        data=insert_data
    )
    print(f"成功插入 {len(insert_data)} 条记录")
    print(f"插入结果: {result}")
except Exception as e:
    print(f"插入数据时出错: {e}")

# 验证数据插入
print("正在验证数据插入...")
try:
    # 查询集合中的数据数量
    stats = client.get_collection_stats(collection_name)
    print(f"集合统计信息: {stats}")
    
    # 进行一个简单的搜索测试
    test_query = "执行一些简单任务"
    test_embedding = model.encode([test_query])
    
    search_results = client.search(
        collection_name=collection_name,
        data=test_embedding.tolist(),
        limit=3,
        output_fields=["text", "source"]
    )
    
    print(f"\n搜索测试 - 查询: '{test_query}'")
    for i, result in enumerate(search_results[0]):
        print(f"结果 {i+1}:")
        print(f"  相似度: {result['distance']:.4f}")
        print(f"  文本: {result['entity']['text'][:100]}...")
        print(f"  来源: {result['entity']['source']}")
        print()
        
except Exception as e:
    print(f"验证时出错: {e}")

print("处理完成！")