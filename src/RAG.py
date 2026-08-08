import os
import re
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
import numpy as np
from openai import OpenAI

class RAGSystem:
    def __init__(self):
        """初始化RAG系统"""
        # 设置路径
        self.model_path = r"your path\models\bge-large-zh-v1.5"
        
        # 加载BGE模型
        print("正在加载BGE模型...")
        self.embedding_model = SentenceTransformer(self.model_path)
        print("BGE模型加载完成！")
        
        # 连接Milvus数据库
        print("正在连接Milvus数据库...")
        self.milvus_client = MilvusClient(uri="http://localhost:19530")#将类的属性与功能赋予到某个对象上
        self.database_name = "your name"
        self.collection_name = "ocr_text_collection"
        
        # 使用数据库
        self.milvus_client.use_database(self.database_name)
        print("Milvus数据库连接成功！")
        
        # 初始化通义千问客户端
        print("正在初始化通义千问客户端...")
        self.llm_client = OpenAI(
            api_key="your api key",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        print("通义千问客户端初始化完成！")
        
    def encode_query(self, query):
        """对用户查询进行向量编码"""
        return self.embedding_model.encode([query])#带中括号是因为在构建知识库的时候原文本也是批量存到列表中转化成向量的，保持格式一致。
    
    def retrieve_relevant_docs(self, query, top_k=3):
        """从Milvus中检索相关文档"""
        # 对查询进行编码
        query_embedding = self.encode_query(query)
        
        # 在Milvus中搜索，search是内置函数
        search_results = self.milvus_client.search(
            collection_name=self.collection_name,#在哪里找，即你已经构建的知识库，里面包含id，text，vector，source
            data=query_embedding.tolist(),#搜索依据，找与data相关的向量，data本来是numpy向量变成list（milvus数据库只能处理list）
            limit=top_k,#召回多少个相关的
            output_fields=["text", "source"]#返回对应数量的原文文本，以及来源..。如果是空列表，则默认返回id
        )
        
        # 提取相关文档
        relevant_docs = []
        for result in search_results[0]:
            relevant_docs.append({
                "text": result['entity']['text'],
                "source": result['entity']['source'],
                "score": result['distance']
            })
        
        return relevant_docs
    
    def generate_prompt(self, query, relevant_docs):
        """生成包含上下文的提示词"""
        context = "\n\n".join([f"文档片段 {i+1}:\n{doc['text']}" #把相关文档中的text组成字符串
                              for i, doc in enumerate(relevant_docs)])
        
        prompt = f"""基于以下文档内容回答用户问题。请仅根据提供的文档内容进行回答，如果文档中没有相关信息，请说明无法从提供的文档中找到答案。

文档内容:
{context}

用户问题: {query}

请基于上述文档内容回答:"""
        
        return prompt
    
    def generate_answer(self, prompt):
        """使用通义千问生成答案"""
        try:
            completion = self.llm_client.chat.completions.create(#创建一个chat对话
                model="qwen3-coder-plus",
                #聊天模式标准输入格式
                messages=[
                    {"role": "system", "content": "你是一个专业的助手，请基于提供的文档内容准确回答用户问题。"},#在聊天中输入系统提示词
                    {"role": "user", "content": prompt},#作为用户输入我们已经设计好的提示词
                ],
                temperature=0.3,#量化大模型的“思维”发散能力，越高，大模型越发散，越不严格按照我们检索出来的参考文本，通常RAG：0.1-0.5
                max_tokens=1000
            )
            return completion.choices[0].message.content#一次请求大模型会输出多个候选答案，这里选择第一个
        except Exception as e:
            return f"生成答案时出错: {str(e)}"
    
    def query(self, user_question, top_k=3, show_sources=True):
        """完整的RAG查询流程"""
        print(f"\n🔍 用户问题: {user_question}")
        print("=" * 50)
        
        # 步骤1: 检索相关文档
        print("📚 正在检索相关文档...")
        relevant_docs = self.retrieve_relevant_docs(user_question, top_k)
        
        if not relevant_docs:
            return "抱歉，没有找到相关的文档内容。"
        
        # 显示检索到的文档
        if show_sources:
            print(f"\n📖 找到 {len(relevant_docs)} 个相关文档片段:")
            for i, doc in enumerate(relevant_docs):
                print(f"\n片段 {i+1} (相似度: {doc['score']:.4f}):")
                print(f"内容: {doc['text'][:150]}...")
                print(f"来源: {doc['source']}")
        
        # 步骤2: 生成提示词
        print(f"\n🤖 正在生成答案...")
        prompt = self.generate_prompt(user_question, relevant_docs)
        
        # 步骤3: 生成答案
        answer = self.generate_answer(prompt)
        
        print(f"\n💡 答案:")
        print(answer)
        
        return {
            "question": user_question,
            "answer": answer,
            "sources": relevant_docs
        }

def main():
    """主函数 - 演示RAG系统"""
    # 初始化RAG系统
    rag = RAGSystem()
    
    print("\n" + "="*60)
    print("🚀 RAG系统初始化完成！")
    print("="*60)
    
    # 测试问题列表
    test_questions = [
        "什么是shell脚本？",
        "Linux系统管理员需要学习什么？",
        "如何在命令行下进行文件管理？",
        "桌面Linux发行版有什么特点？",
        "本书的作用是什么？"
    ]
    
    # 交互式查询
    while True:
        print("\n" + "="*60)
        print("🎯 RAG问答系统")
        print("="*60)
        print("输入 'quit' 或 'exit' 退出")
        print("输入 'test' 运行测试问题")
        print("或直接输入你的问题:")
        
        user_input = input("\n请输入问题: ").strip()
        
        if user_input.lower() in ['quit', 'exit', '退出']:
            print("👋 再见！")
            break
        elif user_input.lower() == 'test':
            print("\n🧪 运行测试问题...")
            for i, question in enumerate(test_questions, 1):
                print(f"\n{'='*20} 测试问题 {i} {'='*20}")
                rag.query(question)
                if i < len(test_questions):
                    input("\n按回车键继续下一个测试问题...")
        elif user_input:
            rag.query(user_input)
        else:
            print("❌ 请输入有效的问题！")

if __name__ == "__main__":
    main()

    