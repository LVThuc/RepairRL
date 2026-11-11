"""Lazy GPU loading shared by generation and training."""
import time
from repairrl.common import BASE_REVISION


def load_model(model_name, revision=BASE_REVISION, adapter=None, quantized=True):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required; run python -m scripts.doctor")
    tokenizer = AutoTokenizer.from_pretrained(adapter or model_name, revision=revision, trust_remote_code=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    options = dict(revision=revision, trust_remote_code=False, device_map={"": 0},
                   torch_dtype=torch.float16, attn_implementation="sdpa")
    if quantized:
        options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(model_name, **options)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
    return model, tokenizer


class Generator:
    def __init__(self, model_name, adapter=None, revision=BASE_REVISION, max_new_tokens=512,
                 max_context=2048, temperature=0.0, seed=42, quantized=True):
        from transformers import set_seed
        set_seed(seed)
        self.model, self.tokenizer = load_model(model_name, revision, adapter, quantized)
        self.model.eval()
        self.max_new_tokens, self.max_context, self.temperature = max_new_tokens, max_context, temperature
        self.model_name, self.adapter, self.seed = model_name, adapter, seed
        self.revision = getattr(self.model.config, "_commit_hash", None) or revision

    def __call__(self, messages):
        import torch
        inputs = self.tokenizer.apply_chat_template(messages, tokenize=True,
                    add_generation_prompt=True, return_tensors="pt", return_dict=True).to(self.model.device)
        length = inputs["input_ids"].shape[-1]
        if length + self.max_new_tokens > self.max_context:
            return {"text": "", "tokens": 0, "seconds": 0.0, "error": "context_overflow"}
        options = {"max_new_tokens": self.max_new_tokens, "do_sample": self.temperature > 0,
                   "pad_token_id": self.tokenizer.pad_token_id}
        if self.temperature > 0:
            options["temperature"] = self.temperature
        torch.cuda.synchronize()
        start = time.monotonic()
        with torch.inference_mode():
            output = self.model.generate(**inputs, **options)
        torch.cuda.synchronize()
        ids = output[0, length:]
        return {"text": self.tokenizer.decode(ids, skip_special_tokens=True), "tokens": len(ids),
                "seconds": time.monotonic() - start}
