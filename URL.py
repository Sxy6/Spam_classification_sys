import re
import string

# 1. 提取 URL 数量的函数
URL_REGEX = re.compile(r'https?://\S+')


def extract_url_count(text):
    """提取文本中 URL 的数量"""
    urls = re.findall(URL_REGEX, text)
    return len(urls)


# 2. 提取特殊符号比例的函数
SPECIAL_SYMBOLS = set(string.punctuation)  # 标点符号集合


def extract_special_symbol_ratio(text):
    """计算文本中特殊符号占总字符数的比例"""
    total_chars = len(text)
    special_chars = sum(1 for char in text if char in SPECIAL_SYMBOLS)

    if total_chars == 0:
        return 0  # 防止除以零
    return special_chars / total_chars


# 3. 处理 CSV 文件并提取特征
def process_csv_with_features(input_file, output_file):
    """读取 CSV 文件，提取 URL 数量和特殊符号比例，并保存到新的 CSV 文件"""
    with open(input_file, mode='r', encoding='utf-8', errors='ignore') as infile, \
            open(output_file, mode='w', encoding='utf-8', newline='') as outfile:

        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # 追加列标题
        header = next(reader)
        header += ['url_count', 'special_symbol_ratio']  # 增加两个新的特征列
        writer.writerow(header)

        # 遍历每一行
        for row in reader:
            cleaned_row = []
            url_count = 0
            special_symbol_ratio = 0

            for cell in row:
                # 移除 NUL 字符
                cell = remove_null_characters(cell)
                # 去除 HTML 标签
                cleaned_cell = clean_html_tags(cell)

                # 提取图片链接并清理
                img_links = extract_img_links(cell)
                cleaned_cell = ' '.join(img_links) if img_links else cleaned_cell

                # 提取 URL 数量和特殊符号比例
                url_count += extract_url_count(cleaned_cell)
                special_symbol_ratio += extract_special_symbol_ratio(cleaned_cell)

                cleaned_row.append(cleaned_cell)

            # 添加特征到行
            cleaned_row += [url_count, special_symbol_ratio]

            # 写入处理后的行
            writer.writerow(cleaned_row)


# 主函数
if __name__ == "__main__":
    input_file = 'input_cleaned.csv'  # 清理后的 CSV 文件路径
    output_file = 'output_with_features.csv'  # 输出特征提取后的文件路径

    # 1. 处理 CSV 文件并提取特征
    process_csv_with_features(input_file, output_file)
    print("CSV 文件处理完成，特征已提取，输出到", output_file)
