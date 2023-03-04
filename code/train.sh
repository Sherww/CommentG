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

python run.py --do_train --do_eval --model_type roberta --model_name_or_path microsoft/codebert-base  --train_filename ../dataset/train.jsonl --dev_filename ../dataset/valid.jsonl --output_dir model/java --max_source_length 128 --max_target_length 32 --beam_size 5 --train_batch_size 128 --eval_batch_size 64 --learning_rate 5e-5 --local_rank 0 --nproc_per_node 6 --num_train_epochs 50  > output/train_java.log 2>&1 


