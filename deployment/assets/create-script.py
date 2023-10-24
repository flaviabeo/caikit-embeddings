import os
import json

MODELS_LIST = json.loads(os.getenv('MODELS_LIST', ["mini", "slate"]))
MODELS_DIR = json.loads(os.getenv('MODELS_DIR', ["sentence-transformers_all-MiniLM-L6-v2","slate.rtvr271M"]))
TRANSFORMERS_CACHE_DIR = os.getenv('TRANSFORMERS_CACHE', "/opt/app-root/src/.cache/huggingface/hub").replace("hub", "")

with open ('../../load-models-config.sh', 'w') as rsh:
    init_models_command = 'cd demo && '
    for model in MODELS_LIST:
        init_models_command += f"cd models/{model} && cd ../../ && cd models/{model}-rr && cd ../../ && cd models/{model}-ss && cd ../../ &&"
    init_models_command += ' cd server && '
    for model_dir in MODELS_DIR:
        init_models_command += f"cd {TRANSFORMERS_CACHE_DIR}sentence_transformers/{model_dir} && cd &&"
    init_models_command += ' cd ' 
    rsh.write(init_models_command)