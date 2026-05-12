import modal
from modal import Volume, Image

app = modal.App("plain-pricer-service")
image = Image.debian_slim().pip_install(
    "huggingface", "torch", "transformers", "bitsandbytes", "accelerate", "peft"
)

secrets = [modal.Secret.from_name("huggingface-secret")]

GPU = "T4"
BASE_MODEL = "meta-llama/Llama-3.2-3B"
PROJECT_NAME = "stock-price-predictor"
HF_USER = "matthewyn"
RUN_NAME = "2026-05-12_06.25.13"
PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"
REVISION = "12f168926d44c3eef47a125eee6b4144989161fc"
FINETUNED_MODEL = f"{HF_USER}/{PROJECT_RUN_NAME}"
CACHE_DIR = "/cache"

MIN_CONTAINERS = 0

PREFIX = "Percentage change is:"
QUESTION = "Given this market summary where the last price was {last_price}, predict the percentage change in price after 7 days. Return a single number representing the percentage change (e.g. +5.2 for +5.2%, -3.1 for -3.1%). Market summary:"

hf_cache_volume = Volume.from_name("hf-hub-cache", create_if_missing=True)

@app.cls(
    image=image.env({"HF_HUB_CACHE": CACHE_DIR}),
    secrets=secrets,
    gpu=GPU,
    timeout=1800,
    min_containers=MIN_CONTAINERS,
    volumes={CACHE_DIR: hf_cache_volume},
)
class Stock_Pricer:
    @modal.enter()
    def setup(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

        # ✅ Use EOS as pad — avoids vocab size change and tokenization shift
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.tokenizer.padding_side = "right"  # ✅ Required for SFTTrainer

        self.base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=quant_config,
            device_map="auto",
        )

        # ✅ No resize needed since we didn't add new tokens
        self.base_model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.fine_tuned_model = PeftModel.from_pretrained(
            self.base_model, FINETUNED_MODEL, revision=REVISION
        )

    @modal.method()
    def price(self, summary: str, last_price: float) -> float:
        import re
        import torch
        from transformers import set_seed

        set_seed(42)
        prompt = f"{QUESTION.format(last_price=last_price)}\n\n{summary}\n\n{PREFIX}"

        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output_ids = self.fine_tuned_model.generate(
                **inputs,
                max_new_tokens=8,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, prompt_len:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        response = response.replace('−', '-')
        pct_change = float(response)
        return pct_change