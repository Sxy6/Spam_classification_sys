from nltk import download
from nltk.tokenize import word_tokenize
import csv
import string
from nltk.corpus import stopwords
# 模块 2: 分词和标准化
def download_nltk_resources():
#下载必要的 NLTK 资源
    download('punkt')
    download('stopwords')


def normalize_text(text):
    """文本标准化：小写化、去除标点符号、去除停用词"""
    # 转小写
    text = text.lower()
    # 分词
    words = word_tokenize(text)
    # 去除标点符号
    words = [word for word in words if word not in string.punctuation]
    # 去除停用词
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return words


def process_text_in_file(input_file):
    """读取文件并进行分词和标准化"""
    tokenized_data = []

    with open(input_file, mode='r', encoding='utf-8', errors='ignore') as infile:
        reader = csv.reader(infile)
        for row in reader:
            for cell in row:
                tokenized_data.append(normalize_text(cell))

    return tokenized_data