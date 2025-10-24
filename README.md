# nvidia
nvidia related scripts



# python

## installation

```bash
cd python
make venv
```

Activate environment

```bash
source .venv/bin/activate
```

Install python packages:
```
pip install -r requirements_full.txt
```

## Image generation

```bash
cd image_generation
python3 diffusion.py
```


# Utilities

Check the size of hugging face caches using disk utility
```bash
du -h ~/.cache/huggingface/ -d 2
```

or using hugging face cli (have to be in the python environment)
```bash
hf cache scan


REPO ID                                    REPO TYPE SIZE ON DISK NB FILES LAST_ACCESSED  LAST_MODIFIED  REFS LOCAL PATH                                                                                   
------------------------------------------ --------- ------------ -------- -------------- -------------- ---- -------------------------------------------------------------------------------------------- 
Qwen/Qwen1.5-1.8B-Chat                     model             3.7G        7 33 minutes ago 33 minutes ago main /home/{USER}/.cache/huggingface/hub/models--Qwen--Qwen1.5-1.8B-Chat                     
TinyLlama/TinyLlama-1.1B-Chat-v1.0         model             2.2G        7 17 minutes ago 41 minutes ago main /home/{USER}/.cache/huggingface/hub/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0         
deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B  model             3.6G        5 17 minutes ago 37 minutes ago main /home/{USER}/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B  
google/gemma-2b-it                         model             5.0G        9 2 minutes ago  2 minutes ago  main /home/{USER}/.cache/huggingface/hub/models--google--gemma-2b-it                         
microsoft/Phi-3-mini-4k-instruct           model             7.6G       12 15 minutes ago 24 minutes ago main /home/{USER}/.cache/huggingface/hub/models--microsoft--Phi-3-mini-4k-instruct           
stabilityai/stable-diffusion-xl-base-1.0   model            14.2G       17 6 days ago     6 days ago     main /home/{USER}/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0   
stablediffusionapi/realistic-vision-v60-b1 model             4.3G       13 2 hours ago    2 hours ago    main /home/{USER}/.cache/huggingface/hub/models--stablediffusionapi--realistic-vision-v60-b1 
```

