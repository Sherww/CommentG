lang=java #programming language
batch_size=64
beam_size=5
source_length=128
target_length=32
data_dir=../dataset
output_dir=model/$lang
train_file=$data_dir/$lang/train.jsonl
dev_file=$data_dir/$lang/valid.jsonl
epochs=50
pretrained_model=microsoft/codebert-base #Roberta: roberta-base


batch_size=16
dev_file=$data_dir/$lang/valid.jsonl
test_file=$data_dir/$lang/test.jsonl
test_model=$output_dir/checkpoint-best-bleu/pytorch_model.bin #checkpoint for test

#python run.py --do_test --model_type roberta --model_name_or_path microsoft/codebert-base --load_model_path $test_model --dev_filename $dev_file --test_filename $test_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --eval_batch_size $batch_size

python run.py --do_test --model_type roberta --model_name_or_path ../../codebert-base --load_model_path model/java/checkpoint-best-bleu/pytorch_model.bin --dev_filename ../dataset/java/valid.jsonl --test_filename ../dataset/java/test.jsonl --output_dir model/java --max_source_length 128 --max_target_length 32 --beam_size 5 --eval_batch_size 64  > output/test_java.log 2>&1
