import jsonlines
from tqdm import tqdm

print('getting data----------')
with open('output.jsonl','r', encoding='utf-8') as f:
    datas=[]
    for data in jsonlines.Reader(f):
        datas.append(data)
with open('D:\\Research\\Codeqg\\CodeQG\\data\\qg_data\\test.jsonl\\test.jsonl','r',encoding='utf-8')as ff:
    datas_init=[]
    for dat in jsonlines.Reader(ff):
        datas_init.append(dat)

print('reading init data-----')
code_cons=[]
for dat in datas_init:
    # question=dat['question']
    # question_type=dat['question_type']
    code=dat['code']
    context=dat['context']
    code_con=code+','+context
    code_cons.append(code_con)

print('reading output data-----')
code_con_outs=[]
for da in datas:
    # question_out=dat['question']
    # question_type_out=dat['question_type']
    code_out=da['code']
    context_out=da['context']
    code_con_out=code_out+','+context_out
    code_con_outs.append(code_con_out)
    
print('starting combine data---------')
outputs=[]
excepts=[]
for data in tqdm(code_cons):
    if data in code_con_outs:
        ids_1= code_cons.index(data)
        ids_2= code_con_outs.index(data)

        #获取初始test数据
        init=datas_init[ids_1]
        quest=init['question']
        quest_type=init['question_type']
        #获取对应id的输出test数据
        output=datas[ids_2]
        #更新输出test数据
        output.update(question=quest)
        output.update(question_type=quest_type)
        outputs.append(output)
        #更新后的数据删除掉
        datas.remove(ids_2)
    else:
        excepts.append(data)
        # print(data)
    print(len(excepts))
print('starting writing-----------')
#output中的内容写下来
with jsonlines.open('combine_test_output.jsonl','w')as fff:
    ff.write_all(outputs)
    # see=datas[0:100]



# from nltk.translate.bleu_score import sentence_bleu
# target = 'this is a small test, which is inster'  # target
# inference = 'this is a small test, which is'  # inference
# # 计算BLEU
# reference= [inference.split()]
# candidate = target.split()
# score1 = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
# score2 = sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0))
# score3 = sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0))
# score4 = sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25))
# reference.clear()
# print('Cumulate 1-gram :%f' \
#       % score1)
# print('Cumulate 2-gram :%f' \
#       % score2)
# print('Cumulate 3-gram :%f' \
#       % score3)
# print('Cumulate 4-gram :%f' \
#       % score4)
