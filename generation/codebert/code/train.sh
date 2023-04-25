lang=java 
lr=5e-5
batch_size=64
beam_size=5
source_length=128
target_length=32
data_dir=../dataset
output_dir=model/$lang
train_file=$data_dir/$lang/train.jsonl
dev_file=$data_dir/$lang/valid.jsonl
epochs=50
pretrained_model=microsoft/codebert-base 

# python run.py --do_train --do_eval --model_type roberta --model_name_or_path $pretrained_model --train_filename $train_file --dev_filename $dev_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --train_batch_size $batch_size --eval_batch_size $batch_size --learning_rate $lr --num_train_epochs $epochs

#python run.py --do_train --do_eval --model_type roberta --model_name_or_path ../../codebert-base  --train_filename ../dataset/java/train.jsonl --dev_filename ../dataset/java/valid.jsonl --output_dir model/java --max_source_length 128 --max_target_length 32 --beam_size 5 --train_batch_size 128 --eval_batch_size 64 --learning_rate 5e-5 --local_rank 0 --nproc_per_node 6 --num_train_epochs 50  > output/train_java.log 2>&1 

python -m torch.distributed.launch --master_port 1234 --nproc_per_node=8 \
         run.py --do_train --do_eval --model_type roberta --model_name_or_path microsoft/codebert-base  --train_filename ../dataset/train.jsonl --dev_filename ../dataset/valid.jsonl --output_dir model/java --max_source_length 128 --max_target_length 32 --beam_size 5 --train_batch_size 128 --eval_batch_size 64 --learning_rate 5e-5 --local_rank 0 --num_train_epochs 50  > output/train_java.log 2>&1 

python how_run.py --do_train --do_eval --model_type roberta --model_name_or_path microsoft/codebert-base  --train_filename ../dataset/how_train.jsonl --dev_filename ../dataset/how_valid.jsonl --output_dir model/how --max_source_length 128 --max_target_length 32 --beam_size 5 --train_batch_size 128 --eval_batch_size 64 --learning_rate 5e-5 --num_train_epochs 50  > output/how_train_java.log 2>&1 

#没有断点续训的
python what_run.py --do_train --do_eval --model_type roberta --model_name_or_path microsoft/codebert-base  --train_filename ../dataset/what_train.jsonl --dev_filename ../dataset/what_valid.jsonl --output_dir model/what_without_duan --max_source_length 128 --max_target_length 32 --beam_size 5 --train_batch_size 128 --eval_batch_size 64 --learning_rate 5e-5 --num_train_epochs 50  > output/what_train_java_without_duan.log 2>&1 
#加上断点续训的
python what_run_duan.py --do_train --do_eval --model_type roberta --model_name_or_path microsoft/codebert-base  --train_filename ../dataset/what_train.jsonl --dev_filename ../dataset/what_valid.jsonl --output_dir model/what --max_source_length 128 --max_target_length 32 --beam_size 5 --train_batch_size 128 --eval_batch_size 64 --learning_rate 5e-5 --num_train_epochs 50  > output/what_train_java.log 2>&1 
#尝试一下能不能续上之前断的地方,epoch 26
python what_run_try.py --do_train --do_eval --model_type roberta --model_name_or_path microsoft/codebert-base  --train_filename ../dataset/what_train.jsonl --dev_filename ../dataset/what_valid.jsonl --output_dir model/what_with_duan_try --max_source_length 128 --max_target_length 32 --beam_size 5 --train_batch_size 128 --eval_batch_size 64 --learning_rate 5e-5 --num_train_epochs 50  > output/what_train_java_with_duan_try.log 2>&1 


python why_run.py --do_train --do_eval --model_type roberta --model_name_or_path microsoft/codebert-base  --train_filename ../dataset/why_train.jsonl --dev_filename ../dataset/why_valid.jsonl --output_dir model/why --max_source_length 128 --max_target_length 32 --beam_size 5 --train_batch_size 128 --eval_batch_size 64 --learning_rate 5e-5 --num_train_epochs 50  > output/why_train_java.log 2>&1 
