# llm-math-reasoning
# Mathematical Reasoning with Large Language Models (LLMs)

## Overview

This project explores the capabilities of Large Language Models (LLMs) in mathematical reasoning tasks. We investigate how LLMs can be leveraged to solve mathematical problems, understand mathematical concepts.

## Key Features

- Implementation of LLM-based mathematical problem-solving
- Evaluation of LLM performance on various mathematical tasks
- Comparison with traditional mathematical reasoning systems

## Getting Started

the llm checkpoints used in this project: mistral-7b, llama2-7b, aritho_math7b

- https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
- https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
- https://huggingface.co/akjindal53244/Arithmo-Mistral-7B

pretrained semantic similarity model: para_minilmL12v2, download link:

- https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- Transformers library
- (Add other relevant libraries)

### Installation

```bash
git clone https://github.com/derby-ding/llm-math-reasoning.git
cd llm-math-reasoning
pip install -r requirements.txt
```

## Usage
###please change the sentence similarity model path in ragenh_math_gsm.py before running
python ragenh_math_gsm.py --promptex sim_cot_sc --infile data/gsm8k_test_formu1.json --model_path your/mistral7b/ --RAG_path data/gsm8k_explanqwenmax2.json.json

##contrastive windows
python selfcorrect_math.py --promptex sim_cot_sc --infile data/gsm8k_main_test.json --model_path your/mistral7b/ --outfile data/gsm8k_contrast.json --shot
_num 3

## Project Structure

```
math-reasoning-llm/
├── data/
├── README.md
└── requirements.txt
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.


## Contact

For any queries, please open an issue or contact [dingkD@zhejianglab.com]
