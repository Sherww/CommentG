import jsonlines
from tqdm import tqdm
def del1():
    # 打开输入文件和输出文件
    with jsonlines.open('how_cg_bart_test_out-20230311-191240.jsonl') as reader, jsonlines.open('how_cg_bart_test_out-20230311-191240_deldup.jsonl', mode='w') as writer:
        # 创建一个集合来跟踪已经读取的行
        seen = []
        i=0
        all_data=[]
        for dat in reader:
            all_data.append(dat)
        for item in tqdm(all_data):
            # 判断数据是否重复
            nee= item['comment']+ item['code']
            if nee in seen:
                i+=1
                # 如果是，则跳过该行并继续读取下一行
                continue
            else:
                # 如果不是，则将行添加到集合中，并写入输出文件
                seen.append(nee)
                writer.write(item)
        print(len(seen))
        print('------------------'+str(i)+'----------------how')
def del2():
    # 打开输入文件和输出文件
    with jsonlines.open('what_cg_bart_test_out-20230317-152351.jsonl') as reader, jsonlines.open('what_cg_bart_test_out-20230317-152351_deldup.jsonl', mode='w') as writer:
        # 创建一个集合来跟踪已经读取的行
        seen = []
        i=0
        all_data=[]
        for dat in reader:
            all_data.append(dat)
        for item in tqdm(all_data):
            # 判断数据是否重复
            nee= item['comment']+ item['code']
            if nee in seen:
                i+=1
                # 如果是，则跳过该行并继续读取下一行
                continue
            else:
                # 如果不是，则将行添加到集合中，并写入输出文件
                seen.append(nee)
                writer.write(item)
        print(len(seen))
        print('------------------'+str(i)+'----------------what')            
def del3():
    # 打开输入文件和输出文件
    with jsonlines.open('why_cg_bart_test_out-20230310-082748.jsonl') as reader, jsonlines.open('why_cg_bart_test_out-20230310-082748_deldup.jsonl', mode='w') as writer:
        # 创建一个集合来跟踪已经读取的行
        seen = []
        i=0
        all_data=[]
        for dat in reader:
            all_data.append(dat)
        for item in tqdm(all_data):
            # 判断数据是否重复
            nee= item['comment']+ item['code']
            if nee in seen:
                i+=1
                # 如果是，则跳过该行并继续读取下一行
                continue
            else:
                # 如果不是，则将行添加到集合中，并写入输出文件
                seen.append(nee)
                writer.write(item)
        print(len(seen))
        print('------------------'+str(i)+'----------------why')
if __name__ =='__main__':
    del1()
    del2()
    del3()