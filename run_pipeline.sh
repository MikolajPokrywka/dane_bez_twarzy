

python train_qlora/inference_vllm.py \
    --input_file example_data/test.short.in.txt \
    --output_file output.vllm.txt \
    --batch_size 32 \
    --adapter ../qlora_output/ || exit 1


python anonymize.py --input output.vllm.txt --output output.regex.txt || exit 1

python train_qlora/generate_synthetic.py --input output.regex.txt --output output.synthetic.txt || exit 1

python evaluate_f1_agnostic.py --pred output.regex.txt --ref example_data/test.short.ref.txt || exit 1  

grep '\[.*\]' output.synthetic.txt | wc -l