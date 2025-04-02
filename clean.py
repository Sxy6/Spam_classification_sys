import csv
import re
import sys
import string
from tqdm import tqdm

csv.field_size_limit(10 * 1024 * 1024)  # 设置最大字段大小为 10MB

# 定义正则表达式来去除 HTML 标签
HTML_TAG_REGEX = re.compile(r'<.*?>')

# 定义正则表达式来提取图片链接
IMG_TAG_REGEX = re.compile(r'<img[^>]+src=["\'](https?://[^"\']+)["\']')

# 定义正则表达式提取 URL 数量的函数
URL_REGEX = re.compile(r'https?://\S+')

def clean_html_tags(text):
    """清除 HTML 标签"""
    return re.sub(HTML_TAG_REGEX, '', text)


def extract_img_links(text):
    """提取图片链接"""
    return re.findall(IMG_TAG_REGEX, text)


def remove_null_characters(text):
    """移除 NUL 字符"""
    return text.replace('\x00', '')


def normalize_text(text):
    """标准化文本格式"""
    # 转换为小写
    text = text.lower()
    # 移除多余的空格
    text = re.sub(r'\s+', ' ', text)
    # 移除特殊字符，但保留基本的标点符号
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    return text.strip()


def clean_file(input_file, cleaned_file):
    """清理文件中的 NUL 字符并保存到新文件"""
    with open(input_file, 'rb') as infile:
        data = infile.read()

    # 移除 NUL 字符
    data = data.replace(b'\x00', b'')

    with open(cleaned_file, 'wb') as outfile:
        outfile.write(data)

def extract_url_count(text):
    """提取文本中 URL 的数量"""
    urls = re.findall(URL_REGEX, text)
    return len(urls)

SPECIAL_SYMBOLS = set(string.punctuation)  # 标点符号集合

def extract_special_symbol_ratio(text):
    """计算文本中特殊符号占总字符数的比例"""
    total_chars = len(text)
    special_chars = sum(1 for char in text if char in SPECIAL_SYMBOLS)

    if total_chars == 0:
        return 0  # 防止除以零
    return special_chars / total_chars


def process_csv(input_file, output_file):
    """读取 CSV 文件，清洗数据并保存到新的 CSV 文件"""
    # 首先计算总行数用于进度条
    with open(input_file, mode='r', encoding='utf-8', errors='ignore') as infile:
        total_rows = sum(1 for _ in csv.reader(infile))
    
    with open(input_file, mode='r', encoding='utf-8', errors='ignore') as infile, \
            open(output_file, mode='w', encoding='utf-8', newline='') as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        # 追加列标题
        header = next(reader)
        header += ['url_count', 'special_symbol_ratio']  # 增加两个新的特征列
        writer.writerow(header)
        
        # 使用tqdm创建进度条
        for row in tqdm(reader, total=total_rows-1, desc="Processing rows"):
            cleaned_row = []
            url_count = 0
            special_symbol_ratio = 0

            for cell in row:
                # 移除 NUL 字符
                cell = remove_null_characters(cell)
                # 去除 HTML 标签
                cleaned_cell = clean_html_tags(cell)
                # 标准化文本
                cleaned_cell = normalize_text(cleaned_cell)

                # 提取图片链接并清理
                img_links = extract_img_links(cell)
                cleaned_cell = cleaned_cell.replace(cell, ' '.join(img_links)) if img_links else cleaned_cell
                
                # 提取URL数量和特殊符号比例
                url_count += extract_url_count(cleaned_cell)
                special_symbol_ratio += extract_special_symbol_ratio(cleaned_cell)

                cleaned_row.append(cleaned_cell)
            
            # 添加特征到行
            cleaned_row += [url_count, special_symbol_ratio]
            # 写入清洗后的行
            writer.writerow(cleaned_row)


if __name__ == "__main__":
    input_file = 'enron_spam_data.csv'  # 输入 CSV 文件路径
    cleaned_file = 'input_cleaned.csv'  # 清理后的文件路径
    output_file = 'output_cleaned.csv'  # 输出 CSV 文件路径

    print("开始处理数据...")
    
    # 首先清理文件中的 NUL 字符
    clean_file(input_file, cleaned_file)

    # 然后处理清理后的文件
    process_csv(cleaned_file, output_file)
    print(f"CSV 文件处理完成，特征已提取，输出到 {output_file}")
