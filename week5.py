import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import pickle
from collections import Counter
import string
from gensim.models import Word2Vec
from dask_ml.model_selection import GridSearchCV as DaskGridSearchCV
from dask.distributed import Client
import multiprocessing
from dask import delayed
import gc


class FeatureEngineer:
    def __init__(self, tfidf_dim=10000, word2vec_dim=100, batch_size=1000):
        self.tfidf_dim = tfidf_dim
        self.word2vec_dim = word2vec_dim
        self.batch_size = batch_size
        self.tfidf_vectorizer = TfidfVectorizer(max_features=tfidf_dim)
        self.word2vec_model = None
        self.stat_scaler = MinMaxScaler()  # 用于统计特征
        self.word2vec_scaler = MinMaxScaler()  # 用于Word2Vec向量

    def load_models(self, tfidf_file, word2vec_file, texts):
        """加载预训练的TF-IDF和Word2Vec模型"""
        # 处理缺失值
        texts = texts.fillna('')

        # 加载TF-IDF向量和词汇表
        with open(tfidf_file, 'rb') as f:
            tfidf_data = pickle.load(f)
            self.tfidf_vectorizer.vocabulary_ = tfidf_data['vocabulary']
            # 重新初始化TF-IDF向量器
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.tfidf_dim,
                vocabulary=tfidf_data['vocabulary']
            )
            # 使用训练数据拟合TF-IDF向量器
            self.tfidf_vectorizer.fit(texts)

        # 加载Word2Vec模型
        self.word2vec_model = Word2Vec.load(word2vec_file)

    @delayed
    def extract_statistical_features_batch(self, texts_batch):
        """批量提取统计特征"""
        features = []
        for text in texts_batch:
            if pd.isna(text) or text == '':
                features.append(np.zeros(4))
                continue
                
            words = text.split()
            word_freq = Counter(words)
            unique_words = len(word_freq)
            total_words = len(words)
            
            special_chars = sum(1 for char in text if char in string.punctuation)
            special_char_ratio = special_chars / len(text) if len(text) > 0 else 0
            
            sender_reputation = 0.0
            sender_reputation += min(len(text) / 1000, 1.0) * 0.3
            sender_reputation += (1 - special_char_ratio) * 0.3
            sender_reputation += (unique_words / total_words) * 0.4 if total_words > 0 else 0
            
            features.append(np.array([
                unique_words,
                total_words,
                special_char_ratio,
                sender_reputation
            ]))
        return np.array(features)
    
    @delayed
    def get_text_vectors_batch(self, texts_batch):
        """批量获取文本向量"""
        # 处理缺失值
        texts_batch = [text if pd.notna(text) else '' for text in texts_batch]
        
        # TF-IDF向量
        tfidf_vectors = self.tfidf_vectorizer.transform(texts_batch)
        
        # Word2Vec向量
        word2vec_vectors = np.zeros((len(texts_batch), self.word2vec_dim))
        for i, text in enumerate(texts_batch):
            if text == '':
                continue
                
            words = text.split()
            word_vectors = []
            for word in words:
                if word in self.word2vec_model.wv:
                    word_vectors.append(self.word2vec_model.wv[word])
            if word_vectors:
                word2vec_vectors[i] = np.mean(word_vectors, axis=0)
        
        return (tfidf_vectors, word2vec_vectors)
    
    @delayed
    def combine_features_batch(self, stat_features, tfidf_vectors, word2vec_vectors):
        """合并特征批次"""
        # 标准化统计特征
        stat_features_scaled = self.stat_scaler.fit_transform(stat_features)
        
        # 标准化Word2Vec向量
        word2vec_vectors_scaled = self.word2vec_scaler.fit_transform(word2vec_vectors)
        
        # 合并所有特征
        return np.hstack([
            stat_features_scaled,
            tfidf_vectors.toarray(),
            word2vec_vectors_scaled
        ])
    
    def create_hybrid_features(self, texts):
        """创建混合特征（使用延迟计算和批处理）"""
        n_samples = len(texts)
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size
        
        # 创建延迟计算任务
        delayed_tasks = []
        for i in range(n_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, n_samples)
            batch = texts[start_idx:end_idx]
            
            # 并行处理统计特征和文本向量
            stat_features = self.extract_statistical_features_batch(batch)
            vectors = self.get_text_vectors_batch(batch)
            
            # 合并特征
            delayed_tasks.append(
                self.combine_features_batch(stat_features, vectors[0], vectors[1])
            )
        
        # 执行所有延迟计算任务
        results = []
        for task in delayed_tasks:
            result = task.compute()
            results.append(result)
            gc.collect()  # 强制垃圾回收
        
        # 合并所有批次的结果
        return np.vstack(results)


def main():
    multiprocessing.freeze_support()

    # 配置Dask客户端
    client = Client(
        n_workers=8,  # 增加工作节点数
        threads_per_worker=1,
        memory_limit='2GB',  # 限制每个工作节点的内存
        dashboard_address=':8788'
    )
    print("Dask 集群信息:", client)

    # 读取数据
    data_file = "tokenized_texts.csv"
    df = pd.read_csv(data_file)

    if 'Message' not in df.columns or 'Spam/Ham' not in df.columns:
        raise ValueError("CSV 文件中需要 'Message' 和 'Spam/Ham' 列")

    # 打印缺失值信息
    print(f"Message列缺失值数量: {df['Message'].isna().sum()}")
    print(f"Spam/Ham列缺失值数量: {df['Spam/Ham'].isna().sum()}")

    # 删除标签中的缺失值
    df = df.dropna(subset=['Spam/Ham'])

    # 特征工程
    feature_engineer = FeatureEngineer(batch_size=500)  # 减小批次大小
    
    # 使用训练数据加载和拟合模型
    feature_engineer.load_models('tfidf_vectors.pkl', 'word2vec_model', df['Message'])

    # 创建混合特征
    X = feature_engineer.create_hybrid_features(df['Message'])
    y = df['Spam/Ham'].values

    # 保存特征和标签
    np.save('X_features.npy', X)
    np.save('y_labels.npy', y)
    print(f"特征维度: {X.shape}")
    print(f"标签维度: {y.shape}")

    # 划分数据集
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print(f"数据集划分完成：训练集 {len(X_train)}, 验证集 {len(X_val)}, 测试集 {len(X_test)}")

    # 训练贝叶斯分类器
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    y_pred_nb = nb_model.predict(X_val)
    nb_acc = accuracy_score(y_val, y_pred_nb)
    print(f"贝叶斯分类器 验证集准确率: {nb_acc:.4f}")

    # 训练SVM模型（使用较小的参数网格）
    svm_model = SVC()
    param_grid = {
        'C': [0.1, 1],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale']
    }

    grid_search = DaskGridSearchCV(
        svm_model,
        param_grid,
        cv=3,
        scoring='accuracy',
        scheduler=client
    )

    grid_search.fit(X_train, y_train)
    best_svm = grid_search.best_estimator_
    y_pred_svm = best_svm.predict(X_val)
    svm_acc = accuracy_score(y_val, y_pred_svm)
    print(f"SVM 最优模型 验证集准确率: {svm_acc:.4f}")
    print(f"最优 SVM 参数: {grid_search.best_params_}")

    # 在测试集上评估最终模型
    final_model = nb_model if nb_acc > svm_acc else best_svm
    y_pred_test = final_model.predict(X_test)
    final_acc = accuracy_score(y_test, y_pred_test)
    print(f"最终选定模型 测试集准确率: {final_acc:.4f}")


if __name__ == '__main__':
    main()