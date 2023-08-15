from transformers import AutoModelForCausalLM, StoppingCriteria, BitsAndBytesConfig
import torch


def load_quantize_model(inf_model, load_4b, load_8b, tokenizer, device_map):
    if load_4b:
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        model = AutoModelForCausalLM.from_pretrained(
            inf_model, pad_token_id=tokenizer.eos_token_id, device_map=device_map, quantization_config=nf4_config)

    elif load_8b:
        model = AutoModelForCausalLM.from_pretrained(
            inf_model, pad_token_id=tokenizer.eos_token_id, device_map=device_map, load_in_8bit=True)

    return model


class MyStoppingCriteria(StoppingCriteria):
    def __init__(self, target_sequence, prompt, tokenizer):
        self.target_sequence = target_sequence
        self.prompt = prompt
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        # Get the generated text as a string
        generated_text = self.tokenizer.decode(input_ids[0])
        generated_text = generated_text.replace(self.prompt, '')
        # Check if the target sequence appears in the generated text
        if self.target_sequence in generated_text:
            return True  # Stop generation
        return False  # Continue generation

    def __len__(self):
        return 1

    def __iter__(self):
        yield self
