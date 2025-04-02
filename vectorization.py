import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import download
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
import os
from tqdm import tqdm
import pickle

# 下载必要的NLTK资源
download('punkt')
download('stopwords')

class TextVectorizer:
    def __init__(self, min_word_freq=2, vector_size=100):
        self.min_word_freq = min_word_freq
        self.vector_size = vector_size
        self.stop_words = set(stopwords.words('english'))
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.word2vec_model = None
        self.tokenized_texts = None
        
    def preprocess_text(self, text):
        """文本预处理：分词、去除停用词和标点符号"""
        # 分词
        tokens = word_tokenize(text.lower())
        # 去除停用词和标点符号
        tokens = [token for token in tokens 
                 if token not in self.stop_words 
                 and token not in string.punctuation]
        return tokens

    def tokenize_dataset(self, input_file, output_file):
        """对数据集进行分词并保存为CSV格式"""
        print("开始分词处理...")
        df = pd.read_csv(input_file)
        
        # 对文本列进行分词
        text_columns = df.select_dtypes(include=['object']).columns
        tokenized_texts = []
        
        # 创建新的DataFrame来存储结果
        result_df = df.copy()
        
        # 添加分词结果列
        for col in text_columns:
            result_df[f'{col}_tokens'] = ''
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing texts"):
            row_tokens = []
            for col in text_columns:
                if pd.notna(row[col]):  # 检查是否为空
                    tokens = self.preprocess_text(str(row[col]))
                    row_tokens.extend(tokens)
                    # 将分词结果存储到对应的列
                    result_df.at[idx, f'{col}_tokens'] = ' '.join(tokens)
            
            tokenized_texts.append(row_tokens)
        
        # 保存为CSV文件
        result_df.to_csv(output_file, index=False, encoding='utf-8')
        
        self.tokenized_texts = tokenized_texts
        print(f"分词完成，结果已保存到 {output_file}")
        return tokenized_texts

    def create_tfidf_vectors(self, output_file):
        """使用TF-IDF创建词向量"""
        print("开始生成TF-IDF向量...")
        if self.tokenized_texts is None:
            raise ValueError("请先运行tokenize_dataset进行分词")
        
        # 将分词结果转换为文本
        texts = [' '.join(tokens) for tokens in self.tokenized_texts]
        
        # 生成TF-IDF向量
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        
        # 保存TF-IDF向量
        with open(output_file, 'wb') as f:
            pickle.dump({
                'matrix': tfidf_matrix,
                'vocabulary': self.tfidf_vectorizer.vocabulary_
            }, f)
        
        print(f"TF-IDF向量已保存到 {output_file}")
        return tfidf_matrix

    def create_word2vec_vectors(self, output_file):
        """使用Word2Vec创建词向量"""
        print("开始训练Word2Vec模型...")
        if self.tokenized_texts is None:
            raise ValueError("请先运行tokenize_dataset进行分词")
        
        # 训练Word2Vec模型
        self.word2vec_model = Word2Vec(
            sentences=self.tokenized_texts,
            vector_size=self.vector_size,
            window=5,
            min_count=self.min_word_freq,
            workers=4
        )
        
        # 保存Word2Vec模型和词向量
        self.word2vec_model.save(output_file)
        print(f"Word2Vec模型和词向量已保存到 {output_file}")
        
        return self.word2vec_model

    def create_embedding_matrix(self, output_file):
        """创建词嵌入矩阵"""
        print("开始创建词嵌入矩阵...")
        if self.word2vec_model is None:
            raise ValueError("请先运行create_word2vec_vectors训练Word2Vec模型")
        
        # 获取词汇表
        vocabulary = self.word2vec_model.wv.index_to_key
        embedding_matrix = np.zeros((len(vocabulary), self.vector_size))
        
        # 构建词嵌入矩阵
        for i, word in enumerate(vocabulary):
            embedding_matrix[i] = self.word2vec_model.wv[word]
        
        # 保存词嵌入矩阵
        with open(output_file, 'wb') as f:
            pickle.dump({
                'matrix': embedding_matrix,
                'vocabulary': vocabulary
            }, f)
        
        print(f"词嵌入矩阵已保存到 {output_file}")
        return embedding_matrix

def main():
    # 设置输入输出文件路径
    input_file = 'output_cleaned.csv'
    tokenized_file = 'tokenized_texts.csv'  # 改为.csv后缀
    tfidf_file = 'tfidf_vectors.pkl'
    word2vec_file = 'word2vec_model'
    embedding_matrix_file = 'embedding_matrix.pkl'
    
    # 创建向量化器实例
    vectorizer = TextVectorizer(min_word_freq=2, vector_size=100)
    
    # 执行分词
    vectorizer.tokenize_dataset(input_file, tokenized_file)
    
    # 生成TF-IDF向量
    vectorizer.create_tfidf_vectors(tfidf_file)
    
    # 生成Word2Vec向量
    vectorizer.create_word2vec_vectors(word2vec_file)
    
    # 创建词嵌入矩阵
    vectorizer.create_embedding_matrix(embedding_matrix_file)
    
    print("所有处理完成！")

if __name__ == "__main__":
    main() 