# IMSM
Interweaving Memories of a Siamese Large Language Model

## Train 
Set hyperparameters about model, dataset, and training in `config.json`.

`nohup python train.py`

For the recommended hyperparameters for fine-tuning llama3 on the GSM8K dataset, please refer to the recommended parameter file.

## Inference
`nohup python model_inference.py --model_path --peft_path  --mode 5 --linear_A_path --linear_B_path  ---device --data_device -max_target_length --test_dataset_path  --output_path  --evaluation_datapath `
