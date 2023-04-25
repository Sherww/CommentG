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

python why_run.py --do_test --model_type roberta --model_name_or_path microsoft/codebert-base --load_model_path model/why/why_checkpoint-best-bleu/why_pytorch_model.bin --dev_filename ../dataset/why_valid.jsonl --test_filename ../dataset/why_test_deldup.jsonl --output_dir model/why --max_source_length 128 --max_target_length 32 --beam_size 5 --eval_batch_size 64  > output/test_why_4.log 2>&1
python why_run.py --do_test --model_type roberta --model_name_or_path microsoft/codebert-base --load_model_path model/why/why_checkpoint-best-bleu/why_pytorch_model.bin --dev_filename ../dataset/why_valid.jsonl --test_filename ../dataset/why_test_deldup.jsonl --output_dir model/why --max_source_length 128 --max_target_length 32 --beam_size 5 --eval_batch_size 64  > output/test_why_3.log 2>&1
python why_run.py --do_test --model_type roberta --model_name_or_path microsoft/codebert-base --load_model_path model/why/why_checkpoint-best-bleu/why_pytorch_model.bin --dev_filename ../dataset/why_valid.jsonl --test_filename ../dataset/why_test_deldup.jsonl --output_dir model/why --max_source_length 128 --max_target_length 32 --beam_size 5 --eval_batch_size 64  > output/test_why_2.log 2>&1
python why_run.py --do_test --model_type roberta --model_name_or_path microsoft/codebert-base --load_model_path model/why/why_checkpoint-best-bleu/why_pytorch_model.bin --dev_filename ../dataset/why_valid.jsonl --test_filename ../dataset/why_test_deldup.jsonl --output_dir model/why --max_source_length 128 --max_target_length 32 --beam_size 5 --eval_batch_size 64  > output/test_why_1.log 2>&1


python what_run_try.py --do_test --model_type roberta --model_name_or_path microsoft/codebert-base --load_model_path model/what_with_duan_try/what_checkpoint-best-bleu/what_pytorch_model.bin --dev_filename ../dataset/what_valid.jsonl --test_filename ../dataset/what_test_deldup.jsonl --output_dir model/what_with_duan_try --max_source_length 128 --max_target_length 32 --beam_size 5 --eval_batch_size 64  > output/test_what_4.log 2>&1
python what_run_try.py --do_test --model_type roberta --model_name_or_path microsoft/codebert-base --load_model_path model/what_with_duan_try/what_checkpoint-best-bleu/what_pytorch_model.bin --dev_filename ../dataset/what_valid.jsonl --test_filename ../dataset/what_test_deldup.jsonl --output_dir model/what_with_duan_try --max_source_length 128 --max_target_length 32 --beam_size 5 --eval_batch_size 64  > output/test_what_3.log 2>&1
python what_run_try.py --do_test --model_type roberta --model_name_or_path microsoft/codebert-base --load_model_path model/what_with_duan_try/what_checkpoint-best-bleu/what_pytorch_model.bin --dev_filename ../dataset/what_valid.jsonl --test_filename ../dataset/what_test_deldup.jsonl --output_dir model/what_with_duan_try --max_source_length 128 --max_target_length 32 --beam_size 5 --eval_batch_size 64  > output/test_what_2.log 2>&1
python what_run_try.py --do_test --model_type roberta --model_name_or_path microsoft/codebert-base --load_model_path model/what_with_duan_try/what_checkpoint-best-bleu/what_pytorch_model.bin --dev_filename ../dataset/what_valid.jsonl --test_filename ../dataset/what_test_deldup.jsonl --output_dir model/what_with_duan_try --max_source_length 128 --max_target_length 32 --beam_size 5 --eval_batch_size 64  > output/test_what_1.log 2>&1


python what_run.py --do_test --model_type roberta --model_name_or_path microsoft/codebert-base --load_model_path model/what/what_checkpoint-best-bleu/what_pytorch_model.bin --dev_filename ../dataset/what_valid.jsonl --test_filename ../dataset/what_test_deldup.jsonl --output_dir model/what --max_source_length 128 --max_target_length 32 --beam_size 5 --eval_batch_size 64  > output/test_what_4.log 2>&1
python what_run.py --do_test --model_type roberta --model_name_or_path microsoft/codebert-base --load_model_path model/what/what_checkpoint-best-bleu/what_pytorch_model.bin --dev_filename ../dataset/what_valid.jsonl --test_filename ../dataset/what_test_deldup.jsonl --output_dir model/what --max_source_length 128 --max_target_length 32 --beam_size 5 --eval_batch_size 64  > output/test_what_3.log 2>&1
python what_run.py --do_test --model_type roberta --model_name_or_path microsoft/codebert-base --load_model_path model/what/what_checkpoint-best-bleu/what_pytorch_model.bin --dev_filename ../dataset/what_valid.jsonl --test_filename ../dataset/what_test_deldup.jsonl --output_dir model/what --max_source_length 128 --max_target_length 32 --beam_size 5 --eval_batch_size 64  > output/test_what_2.log 2>&1
python what_run.py --do_test --model_type roberta --model_name_or_path microsoft/codebert-base --load_model_path model/what/what_checkpoint-best-bleu/what_pytorch_model.bin --dev_filename ../dataset/what_valid.jsonl --test_filename ../dataset/what_test_deldup.jsonl --output_dir model/what --max_source_length 128 --max_target_length 32 --beam_size 5 --eval_batch_size 64  > output/test_what_1.log 2>&1

python how_run.py --do_test --model_type roberta --model_name_or_path microsoft/codebert-base --load_model_path model/how/how_checkpoint-best-bleu/how_pytorch_model.bin --dev_filename ../dataset/how_valid.jsonl --test_filename ../dataset/how_test_deldup.jsonl --output_dir model/how --max_source_length 128 --max_target_length 32 --beam_size 5 --eval_batch_size 64  > output/test_how_4.log 2>&1
python how_run.py --do_test --model_type roberta --model_name_or_path microsoft/codebert-base --load_model_path model/how/how_checkpoint-best-bleu/how_pytorch_model.bin --dev_filename ../dataset/how_valid.jsonl --test_filename ../dataset/how_test_deldup.jsonl --output_dir model/how --max_source_length 128 --max_target_length 32 --beam_size 5 --eval_batch_size 64  > output/test_how_3.log 2>&1
python how_run.py --do_test --model_type roberta --model_name_or_path microsoft/codebert-base --load_model_path model/how/how_checkpoint-best-bleu/how_pytorch_model.bin --dev_filename ../dataset/how_valid.jsonl --test_filename ../dataset/how_test_deldup.jsonl --output_dir model/how --max_source_length 128 --max_target_length 32 --beam_size 5 --eval_batch_size 64  > output/test_how_2.log 2>&1
python how_run.py --do_test --model_type roberta --model_name_or_path microsoft/codebert-base --load_model_path model/how/how_checkpoint-best-bleu/how_pytorch_model.bin --dev_filename ../dataset/how_valid.jsonl --test_filename ../dataset/how_test_deldup.jsonl --output_dir model/how --max_source_length 128 --max_target_length 32 --beam_size 5 --eval_batch_size 64  > output/test_how_1.log 2>&1

