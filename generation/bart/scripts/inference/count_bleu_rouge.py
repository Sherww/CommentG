# import jsonlines
# from nltk.translate.bleu_score import corpus_bleu
# from rouge import Rouge
# from meteor import meteor_score

# # 读取数据
# data = []
# with jsonlines.open('why_cg_bart_test_out-20230310-082748.jsonl') as f:
#     for line in f:
#         data.append(line)

# # 提取参考文本和目标文本
# references = []
# targets = []
# for example in data:
#     references.append([example['comment']])
#     targets.append(example['prediction'])

# # 计算BLEU指标
# weights = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)]
# bleu_scores = []
# for i in range(4):
#     bleu_scores.append(corpus_bleu(references, targets, weights=weights[i]))

# # 计算ROUGE-L指标
# rouge_scores = Rouge().get_scores(targets, references, avg=True)

# # 计算METEOR指标
# meteor_scores = meteor_score(targets, references)

# # 输出结果
# print('BLEU-1:', bleu_scores[0])
# print('BLEU-2:', bleu_scores[1])
# print('BLEU-3:', bleu_scores[2])
# print('BLEU-4:', bleu_scores[3])
# print('ROUGE-L:', rouge_scores['rouge-l']['f'])
# print('METEOR:', meteor_scores)


import jsonlines
from nltk.translate import meteor_score
from rouge import Rouge
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu,SmoothingFunction



# 打开 JSONL 文件并读取数据
data=[]
# with jsonlines.open('why_cg_bart_test_out-20230310-082748_deldup.jsonl') as f:
with jsonlines.open('how_cg_bart_test_out-20230311-191240_deldup.jsonl') as f:
# with jsonlines.open('what_cg_bart_test_out-20230317-152351_deldup.jsonl') as f:
    for line in f:
        data.append(line)

# 提取参考文本和目标文本
references = []
targets = []
for example in tqdm(data):
    references.append([example['comment'].replace('<sep>','').replace('@&',',').replace('//','').replace('/*','').replace('*/','')])
    targets.append(''.join(example['prediction']).replace('<sep>','').replace('@&',',').replace('//','').replace('/*','').replace('*/',''))
    
# 初始化 ROUGE 和 SmoothingFunction 对象
rouge = Rouge()
smoothing_function = SmoothingFunction().method1

# 初始化 BLEU 和 Meteor 计算的变量
bleu1_scores = []
bleu2_scores = []
bleu3_scores = []
bleu4_scores = []
meteor_scores = []
rouge_scores = []
bleu_scores = []

# 计算BLEU指标
weights = [(1, 0, 0, 0), (0.5, 0.5, 0, 0), (0.33, 0.33, 0.33, 0), (0.25, 0.25, 0.25, 0.25)]

# for example in data:
#     nee=example.split(',')
#     references.append([nee[1].replace('<sep>','').replace('@&',',').replace('//','').replace('/*','').replace('*/','')])
#     targets.append(nee[0].replace('<sep>','').replace('@&',',').replace('//','').replace('/*','').replace('*/',''))


for i in range(4):
    bleu_scores.append(corpus_bleu(references, targets, weights=weights[i]))

# 迭代 JSONL 文件中的每一行
for example in tqdm(data):
    try:
        reference = example['comment'].replace('<sep>','').replace('@&',',').replace('//','').replace('/*','').replace('*/','')
        target =''.join(example['prediction']).replace('<sep>','').replace('@&',',').replace('//','').replace('/*','').replace('*/','')
        
        # target = ''.join(target)
        reference_tokens = reference.split()
        target_tokens = target.split()
        
    
        # 计算 ROUGE-L 值
        rouge_score = rouge.get_scores(target, reference)[0]['rouge-l']['f']
        
        # 添加 ROUGE-L 值到列表中
        rouge_scores.append(rouge_score)
        
        # 计算 Meteor 值
        meteor_scoree = meteor_score.meteor_score([reference_tokens], target_tokens)
        
        # 添加 Meteor 值到列表中
        meteor_scores.append(meteor_scoree)
    except:
        pass

# 打印平均值
print('BLEU-1:', bleu_scores[0]*100)
print('BLEU-2:', bleu_scores[1]*100)
print('BLEU-3:', bleu_scores[2]*100)
print('BLEU-4:', bleu_scores[3]*100)
# print('BLEU1:', sum(bleu1_scores) / len(bleu1_scores))
# print('BLEU2:', sum(bleu2_scores) / len(bleu2_scores))
# print('BLEU3:', sum(bleu3_scores) / len(bleu3_scores))
# print('BLEU4:', sum(bleu4_scores) / len(bleu4_scores))
print('ROUGE-L:', (sum(rouge_scores) / len(rouge_scores))*100)
print('Meteor:', (sum(meteor_scores) / len(meteor_scores))*100)
