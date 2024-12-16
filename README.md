# IMSM
Interweaving Memories of a Siamese Large Language Model

## Train 
Set parameters about model, dataset, and training in `config.json`.

`nohup python train.py`

## Inference
`nohup python model_inference.py --model_path --peft_path  --mode 5 --linear_A_path --linear_B_path  ---device --data_device -max_target_length --test_dataset_path  --output_path  --evaluation_datapath `
