`We are updating the code for the submission` <br>
# submission_93_sec
<img width="1000" alt="edgemmeval" src="https://github.com/user-attachments/assets/d12996bf-bc05-4c74-a32f-bb128a5096ab" />

Code, data, and experiments for our paper "Multi-Modal LLMs at the Edge: A Comparative Study Against Specialized Edge Pipelines". Specifically, we organized the repo in two main parts: (1) five different subdirectories, corresponding to the five tasks considered in the first part (`OC`, `ASR`, `IC`, `AR`, `HD`), and (2) one subdirectory with the source code and experiments for `EdgeMMEval`.

## [PART 1 - 5 tasks] Getting models for reproducibility 
As we used HuggingFace for downloading the MMLLMs, to also successfully download the models (as you run the code), you need to have a [HuggingFace token](https://huggingface.co/docs/hub/en/security-tokens#) configured, and saved under `~/.cache/huggingface/token` on your machine. Subsequently, you need to accept the licence for each of the models. For convenience, once you are logged into your HuggingFace account, simply navigate to each of the following links, and accept the terms and conditions for using the particular models:
1. `gemma3n-e2b`: [https://huggingface.co/google/gemma-3n-E2B-it](https://huggingface.co/google/gemma-3n-E2B-it)
2. `gemma3n-e4b`: [https://huggingface.co/google/gemma-3n-E4B-it](https://huggingface.co/google/gemma-3n-E4B-it)
3. `blip2`: [https://huggingface.co/Salesforce/blip2-flan-t5-xl](https://huggingface.co/Salesforce/blip2-flan-t5-xl)
4. `qwen3-vl-2b`: [https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
5. `ministral-3b`: [https://huggingface.co/mistralai/Ministral-3-3B-Base-2512](https://huggingface.co/mistralai/Ministral-3-3B-Base-2512)

For the classic edge models, we either used `torchvision` models which should be already installed in the virtual environments, or alternatively, we provide the models as a file directly for the corresponding use-case(s) (rest assured, all instructions are provided in the adjacent README.md files).


## [PART 2 - `EdgeMMEval`] Reproducibility and testing
