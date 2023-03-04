from rouge import FilesRouge
from rouge import Rouge
import re


generate=[]
refer=[]
with open('test_0_1.output','r',encoding='utf-8')as f:
    datas=f.readlines()
    for dat in datas:
        dat=dat.replace('.','@').replace('\n','')
        dat=dat+' '+'.'
        data=re.sub(r'^\d+','',dat)
        generate.append(data.replace('	',''))
with open('test_0_1.gold','r',encoding='utf-8')as ff:
    datasf=ff.readlines()
    for datf in datasf:
        datf=datf.replace('.','@').replace('\n','')
        datf=datf+' '+'.'
        dataf=re.sub(r'^\d+','',datf)
        refer.append(dataf.replace('	',''))
rouge = Rouge()
print(rouge.get_scores(generate, refer, avg = True))
files_rouge = FilesRouge()
scores = files_rouge.get_scores('test_0_1.output', 'test_0_1.gold', avg=True)

# files_rouge = FilesRouge()
# scores = files_rouge.get_scores('test_0_1.output', 'test_0_1.gold')

# scoress = files_rouge.get_scores('test_0_1.output', 'test_0_1.gold', avg=True)

