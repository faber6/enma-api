import os
import uvicorn
import torch
import yaml

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline, AutoTokenizer

from models import Completion

app = FastAPI(
    title="Enma Inference API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

args = {
    "device": int(os.getenv("INFERENCE_DEVICE", 0)),
    "config": os.getenv("GATEWAY_CONF", "config.yaml"),
    "port": int(os.getenv("INFERENCE_PORT", 8080))
}

with open(args["config"], "r") as f:
    config = yaml.safe_load(f)
    
inf_model = None
for engine in config["models"].keys():
    inf_model = config["models"][engine]["path"]
# model = pipeline("text-generation", model=inf_model, device=args["device"], load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, torch_dtype=torch.float16)
model = pipeline("text-generation", model=inf_model, device=args["device"], torch_dtype=torch.float16)

tokenizer_with_prefix_space = AutoTokenizer.from_pretrained(inf_model, add_prefix_space=True)

def get_tokens_as_list(word_list):
    "Converts a sequence of words into a list of tokens"
    tokens_list = []
    for word in word_list:
        tokenized_word = tokenizer_with_prefix_space([word], add_special_tokens=False).input_ids[0]
        tokens_list.append(tokenized_word)
    return tokens_list

@app.post("/completion")
async def completion(completion: Completion):
    try:
        return model(
            completion.prompt,
            max_new_tokens=completion.max_new_tokens,
            temperature=completion.temperature,
            top_p=completion.top_p,
            top_k=completion.top_k,
            repetition_penalty=completion.repetition_penalty,
            do_sample=completion.do_sample,
            penalty_alpha=completion.penalty_alpha,
            num_return_sequences=completion.num_return_sequences,
            stop_sequence=completion.stop_sequence
        )
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(
        "inference:app",
        host="0.0.0.0",
        port=args["port"],
    )
