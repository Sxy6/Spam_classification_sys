from gensim.models import Word2Vec

# 模块 3: 词向量化（Word2Vec）
def train_word2vec_model(tokenized_data, vector_size=100, window=5, min_count=1):
    """训练 Word2Vec 模型"""
    model = Word2Vec(sentences=tokenized_data, vector_size=vector_size, window=window, min_count=min_count, workers=4)
    return model


def save_word2vec_model(model, model_path):
    """保存训练好的 Word2Vec 模型"""
    model.save(model_path)


def load_word2vec_model(model_path):
    """加载训练好的 Word2Vec 模型"""
    return Word2Vec.load(model_path)
