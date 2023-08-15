import os
import uvicorn
import yaml

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from models import Completion
from utils import get_tokens_as_list
from transformers import AutoTokenizer


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
    "config": os.getenv("GATEWAY_CONF", "config.yaml"),
    "port": int(os.getenv("INFERENCE_PORT", 8080))
}

with open(args["config"], "r") as f:
    config = yaml.safe_load(f)

inf_model = None
for engine in config["models"].keys():
    inf_model = config["models"][engine]["path"]

tokenizer_with_prefix_space = AutoTokenizer.from_pretrained(
    inf_model, add_prefix_space=True)
tokenizer = AutoTokenizer.from_pretrained(inf_model)


if config['load-in-4b'] or config['load-in-8b']:
    from quantize import load_quantize_model, MyStoppingCriteria

    model = load_quantize_model(
        inf_model, config['load-in-4b'], config['load-in-8b'], tokenizer, config['device-map'])

else:
    from transformers import pipeline
    import torch

    model = pipeline("text-generation", model=inf_model, pad_token_id=tokenizer.eos_token_id,
                     device_map=config['device-map'], torch_dtype=torch.float16)


@app.post("/completion")
async def completion(completion: Completion):
    try:
        if config['load-in-4b'] or config['load-in-8b']:
            inputs = tokenizer(completion.prompt,
                               return_tensors="pt").to('cuda')

            output = model.generate(
                **inputs,
                max_new_tokens=completion.max_new_tokens,
                temperature=completion.temperature,
                top_p=completion.top_p,
                top_k=completion.top_k,
                repetition_penalty=completion.repetition_penalty,
                do_sample=completion.do_sample,
                penalty_alpha=completion.penalty_alpha,
                num_return_sequences=completion.num_return_sequences,
                stopping_criteria=MyStoppingCriteria(
                    completion.stop_sequence, completion.prompt, tokenizer),
                bad_words_ids=get_tokens_as_list(
                    completion.bad_words, tokenizer_with_prefix_space),
                eos_token_id=completion.eos_token_id
            )
            return [{'generated_text': tokenizer.decode(output[0], skip_special_tokens=True)}]

        else:
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
                stop_sequence=completion.stop_sequence,
                bad_words_ids=get_tokens_as_list(
                    completion.bad_words, tokenizer_with_prefix_space),
                eos_token_id=completion.eos_token_id
            )

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(
        "inference:app",
        host="0.0.0.0",
        port=args["port"],
    )
