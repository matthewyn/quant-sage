import modal
from modal import Volume, Image

app = modal.App("pricer-service")
image = Image.debian_slim().pip_install(
    "huggingface", "torch", "transformers", "bitsandbytes", "accelerate", "peft", "sentence-transformers", "pinecone"
)

secrets = [modal.Secret.from_name("huggingface-secret"), modal.Secret.from_name("pinecone-secret")]

GPU = "T4"
BASE_MODEL = "meta-llama/Llama-3.2-3B"
PROJECT_NAME = "stock-price-predictor"
HF_USER = "matthewyn"
RUN_NAME = "2026-04-26_12.41.46"
PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}"
REVISION = "5abc764ed10059e9cab0aeafb204786fa26062a5"
FINETUNED_MODEL = f"{HF_USER}/{PROJECT_RUN_NAME}"
CACHE_DIR = "/cache"

MIN_CONTAINERS = 0

PREFIX = "Percentage change is"
QUESTION = "Given this market summary where the last price was {last_price}, predict the percentage change in price after 30 days. Return a single number representing the percentage change (e.g. 5.2 for +5.2%, -3.1 for -3.1%). Market summary:"

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
        from sentence_transformers import SentenceTransformer
        from pinecone import Pinecone
        import os

        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

        index_name = "stocks"
        self.index = pc.Index(index_name)

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        self.base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, quantization_config=quant_config, device_map="auto"
        )
        self.base_model.resize_token_embeddings(len(self.tokenizer))
        self.fine_tuned_model = PeftModel.from_pretrained(
            self.base_model, FINETUNED_MODEL, revision=REVISION
        )
        self.encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    def vector(self, summary):
        return self.encoder.encode(summary)
    
    def find_similars(self, summary):
        vec = self.vector(summary).astype(float).tolist()
        
        results = self.index.query(vector=vec, top_k=3, include_metadata=True)
        
        documents = [match["metadata"]["document"] for match in results["matches"]]
        prices = [
            float(match["metadata"]["return_percent"])
            for match in results["matches"]
        ]
        
        return documents, prices
    
    def make_context(self, summary):
        context = "For context, here are some similar market summaries and their corresponding return percentages:\n\n"
        similars, prices = self.find_similars(summary)
        for (similar, return_percent) in zip(similars, prices):
            context += f"Market summary: \n{similar}\nReturn percentage: {return_percent}\n\n"
        return context

    @modal.method()
    def price(self, summary: str, last_price: float) -> float:
        import re
        import torch
        from transformers import set_seed

        context = self.make_context(summary)

        set_seed(42)
        prompt = f"{QUESTION.format(last_price=last_price)}\n\n{summary}\n\n{context}{PREFIX}"

        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.fine_tuned_model.generate(
                **inputs,
                max_new_tokens=8,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = outputs[0, prompt_len:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return response.strip()