import csv
import string
from statistics import median, mean
from numpy import percentile
import glob
from tqdm import tqdm

csv.field_size_limit(500 * 1024 * 1024)

def merge_csv_files(file_paths):
    # 合并多个 CSV 文件
    merged_data = []
    for file_path in file_paths:
        with open(file_path, 'r',encoding='utf-8') as file:
            reader = csv.reader(file)
            # next(reader)  # 跳过表头
            for row in reader:
                merged_data.append(row)
    return merged_data
def get_token_counts(column_data):
    """
    统计指定列的所有token数目以及unique token 数目，
    返回总 token 数量和 unique token 数量
    """
    tokens = []
    unique_tokens = set()
    punctuations = string.punctuation  # 获取所有标点符号

    for text in tqdm(column_data):
        # 去除标点符号并分割单词
        words = text.translate( str.maketrans(punctuations, " " * len(punctuations))).split()
        tokens.extend(words)
        
        unique_tokens.update(words)
    return len(tokens), len(unique_tokens)


def get_token_length_stats(column_data):
    """
    统计指定列的长度的四分之一位值，中位数，四分之三位值，平均值，
    返回这些统计量的列表
    """
    lengths = []
    punctuations = string.punctuation  # 获取所有标点符号
    translator = str.maketrans(punctuations, " " * len(punctuations))

    for text in tqdm(column_data):        
    
        # 将所有标点符号替换为空格
        text_without_punctuations = text.translate(translator)
        # text_without_punctuations =text_without_punctuations.strip()
        # text = text.translate(str.maketrans('', '', string.punctuation)).split()
        lengths.append(len(text_without_punctuations.strip().split()))
    # print(lengths)
        # print(len(text))
    q1 = percentile(lengths, 25)
    median_value = median(lengths)
    q3 = percentile(lengths, 75)
    avg = mean(lengths)
    return [q1, median_value, q3, avg]


def get_token_stats(merged_data):
    """
    统计指定列的信息和 token 统计量，以及重复率，
    返回结果字典
    """
    comment_data = []
    code_data = []
    # with open(file_path, 'r', newline='', encoding='utf-8') as f:
        # reader = csv.reader(f)
        # next(reader)  # 跳过标题行
    for row in tqdm(merged_data):
            comment_data.append(row[0].replace('<sep>','').replace('@&',','))  # comment列为第5列
            code_data.append(row[2].replace('<sep>','').replace('@&',','))  # code列为第7列
            # print(comment_data)
    print('counting---------comment_stats-------code_stats')
    # 获取 comment 和 code 列的统计量
    comment_stats = get_token_length_stats(comment_data)
    code_stats = get_token_length_stats(code_data)
    
    print('counting---------comment_token_count-------code_token_count')
    # 计算 comment 和 code 列的 token 统计量
    comment_token_count, comment_unique_token_count = get_token_counts(comment_data)
    code_token_count, code_unique_token_count = get_token_counts(code_data)

    print('counting---------comment_avg_token_count-------code_avg_token_count')
    # 计算 comment 和 code 列的平均 token 数量和平均 unique token 数量
    comment_avg_token_count = comment_token_count / len(comment_data)
    comment_avg_unique_token_count = comment_unique_token_count / len(comment_data)
    code_avg_token_count = code_token_count / len(code_data)
    code_avg_unique_token_count = code_unique_token_count / len(code_data)

    print('counting---------duplicate_count-------duplicate_rate')
    # 计算 comment 和 code 列合并在一起后的重复率
    combined_data=[]
    for a in range(len(comment_data)):
        combined_data.append(comment_data[a]+code_data[a])
    # repeat=[]
    # duplicate_count=0
    # for a in tqdm(combined_data):
    #     if a in repeat:
    #         duplicate_count+=1
    #     else:
    #         repeat.append(a)
            
    set_len = len(set(combined_data))  # 集合长度即为不重复元素数
    duplicate_rate = 1 - set_len / len(combined_data)
    
    # total_count, unique_count = get_token_counts(combined_data)
    # duplicate_count = total_count - unique_count
    # duplicate_rate = duplicate_count / len(combined_data)
    print('repeat-----------------num-----------'+str(set_len))
    # 组织结果字典并返回
    result = {
        'comment_stats': comment_stats,
        'code_stats': code_stats,
        'comment_token_count': comment_token_count,
        'comment_unique_token_count': comment_unique_token_count,
        'code_token_count': code_token_count,
        'code_unique_token_count': code_unique_token_count,
        'comment_avg_token_count': comment_avg_token_count,
        'comment_avg_unique_token_count': comment_avg_unique_token_count,
        'code_avg_token_count': code_avg_token_count,
        'code_avg_unique_token_count': code_avg_unique_token_count,
        'duplicate_rate': duplicate_rate
    }
    return result


if __name__ == '__main__':
    files=glob.glob('../cleandata/Inline-*_not_noisy.csv')

    # files=glob.glob('D:\\Research\\Inline_generation\\dataset\\Inline-0-30.csv')
    merged_data=merge_csv_files(files)
    # file_path = 'Inline-0-30.csv'
    result = get_token_stats(merged_data)
    print(result)
