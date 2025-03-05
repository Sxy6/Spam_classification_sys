from collections import Counter

import clean
import Tokenization
import Word2Vec

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('punkt_tab')  # 下载缺失的资源

# 主函数
if __name__ == "__main__":
    input_file = '/enron_spam_data.csv'  # 清理后的 CSV 文件路径
    cleaned_file = 'input_cleaned.csv'
    output_file = '.venv/output_tokenized.csv'  # 输出分词后的文件路径
    #
    # output_with_features = 'output_with_features.csv'  # 输出特征提取后的文件路径
    # URL.process_csv_with_features(input_file, output_with_features)
    # print("CSV 文件处理完成，特征已提取，输出到", output_with_features)

    word2vec_model_path = '.venv/word2vec.model'  # Word2Vec 模型保存路径
    clean.clean_file(input_file, cleaned_file)
    # 1. 处理 CSV 文件，清理 HTML 标签和图片链接
    clean.process_csv(cleaned_file, output_file)
    print("CSV 文件清理完成，输出到", output_file)

# 2. 下载 NLTK 资源
#     Tokenization.download_nltk_resources()

    # 3. 分词和标准化处理
    tokenized_data = Tokenization.process_text_in_file(output_file)
    print("分词和标准化处理完成。")

    # 4. 训练 Word2Vec 模型
    model = Word2Vec.train_word2vec_model(tokenized_data)
    print("Word2Vec 模型训练完成。")

    # 5. 保存 Word2Vec 模型
    Word2Vec.save_word2vec_model(model, word2vec_model_path)
    print("Word2Vec 模型已保存。")

    # 6. 加载并使用模型（示例）
    loaded_model = Word2Vec.load_word2vec_model(word2vec_model_path)
    print("模型加载完成，示例词向量：", loaded_model.wv['hello'])  # 查看某个词的词向量
