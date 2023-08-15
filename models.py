from pydantic import BaseModel
from typing import Optional, List


class Engine(BaseModel):
    name: str
    author: str
    description: str


class Engines(BaseModel):
    engines: List[Engine]


class Completion(BaseModel):
    prompt: str
    engine: Optional[str] = None
    max_new_tokens: Optional[int] = 20
    eos_token_id: Optional[int] = 198
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    typical_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    do_sample: Optional[bool] = True
    penalty_alpha: Optional[float] = None
    num_return_sequences: Optional[int] = 1
    stop_sequence: Optional[str] = "."
    bad_words: Optional[list] = None
    sequence_bias: Optional[dict] = None


class Feedback(BaseModel):
    prompt: str
    chosen: str
    rejected: str
