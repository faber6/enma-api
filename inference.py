import os
import uvicorn
import torch
import yaml

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from transformers import AutoTokenizer

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

if config['load-in-4b'] or config['load-in-8b']:
    from transformers import AutoConfig, AutoModelForCausalLM, StoppingCriteria, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(inf_model)

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    auto_config = AutoConfig.from_pretrained(inf_model)

    if config['load-in-4b']:
        model = AutoModelForCausalLM.from_pretrained(
            inf_model, pad_token_id=tokenizer.eos_token_id, device_map=config['device-map'], quantization_config=nf4_config)

    elif config['load-in-8b']:
        model = AutoModelForCausalLM.from_pretrained(
            inf_model, pad_token_id=tokenizer.eos_token_id, device_map=config['device-map'], load_in_8bit=True)

    class MyStoppingCriteria(StoppingCriteria):
        def __init__(self, target_sequence, prompt):
            self.target_sequence = target_sequence
            self.prompt = prompt

        def __call__(self, input_ids, scores, **kwargs):
            # Get the generated text as a string
            generated_text = tokenizer.decode(input_ids[0])
            generated_text = generated_text.replace(self.prompt, '')
            # Check if the target sequence appears in the generated text
            if self.target_sequence in generated_text:
                return True  # Stop generation
            return False  # Continue generation

        def __len__(self):
            return 1

        def __iter__(self):
            yield self

else:
    from transformers import pipeline

    model = pipeline("text-generation", model=inf_model, pad_token_id=tokenizer.eos_token_id,
                     device_map=config['device-map'], torch_dtype=torch.float16)


def get_tokens_as_list(word_list):
    "Converts a sequence of words into a list of tokens"
    if word_list:
        tokens_list = []
        for word in word_list:
            tokenized_word = tokenizer_with_prefix_space(
                [word], add_special_tokens=False).input_ids[0]
            tokens_list.append(tokenized_word)
        return tokens_list
    return None


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
                    completion.stop_sequence, completion.prompt),
                bad_words_ids=get_tokens_as_list(completion.bad_words),
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
                bad_words_ids=get_tokens_as_list(completion.bad_words),
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
