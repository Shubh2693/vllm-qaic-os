# MMMU accuracy evaluaiton of LLM models

## Command to evaluate InternVL model on QAIC Pytorch Eager mode stack

```bash
python test_vlm_accuracy.py --model-name OpenGVLab/InternVL3_5-8B-Instruct --dataset-name MMMU/MMMU --ctx-len 4096 --generation-len 2 --nsamples 100 --tp-size 4 --output-file preds.json   --log-file run.log --debug true
```

MMMU Dataset QAIC Accuracy Script (vLLM)

options:

-   -h, --help
`show this help message and exit`
- --model-name MODEL_NAME, --model_name MODEL_NAME
`HF model id or local path`
- --dataset-name DATASET_NAME, --dataset_name DATASET_NAME  
`HF dataset path for MMMU`
- --ctx-len CTX_LEN, --ctx_len CTX_LEN
`Max model context length`
- --generation-len GENERATION_LEN, --generation_len GENERATION_LEN
`Max tokens to generate per prompt`
- --output-file OUTPUT_FILE, --output_file OUTPUT_FILE
`Where to write per-sample predictions`
- --log-file LOG_FILE, --log_file LOG_FILE
`Where to write logs`
- --nsamples NSAMPLES
`Number of validation samples to evaluate (-1 = all)`
- --debug DEBUG
`Enable verbose debug logs`
- --tp-size TP_SIZE
`Tensor parallelism size`
- --kv-cache-bytes KV_CACHE_BYTES
`KV cache memory in bytes (per worker)`
- --qaic-sdpa-decode QAIC_SDPA_DECODE
`Enable SDPA Decode path in vLLM via QAIC`

## Accuracies
---
- Metric: Accuracy
    -   Defination: MMMU (Massive Multi-discipline Multimodal Understanding) evaluates multimodal models on tasks
    requiring advanced perception and reasoning using textual and visual information.
    - Accuracy = Number of correct answers / Total number of questions
- OpenGVLab/InternVL3_5-8B-Instruct
    - Dataset: MMMU
    - No of samples: 900
    - Precision: Weights FP16 + KVFP16
