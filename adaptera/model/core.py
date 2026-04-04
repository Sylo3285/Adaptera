"""
Hugging Face + LM Studio model implementations with PEFT/QLoRA support.
Optional persistent memory via VectorDB for retrieval-augmented prompts.
Uses a small transformer (MiniLM) for automatic embeddings if memory is provided.
"""

import os
import torch
import warnings
import requests

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from peft import PeftModel
from typing import Any, List, Optional

from adaptera.memory.core import VectorDB

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# ---------------------------
# BitsAndBytes optional import
# ---------------------------
try:
    from transformers import BitsAndBytesConfig
    _BNB_AVAILABLE = True
except ImportError:
    _BNB_AVAILABLE = False


# =========================================================
# SHARED MEMORY LAYER (DO NOT DUPLICATE ACROSS MODELS)
# =========================================================
class Memory_Helper:

    @torch.inference_mode()
    def _embed_text(self, text: str) -> torch.Tensor:
        if self.memory is None:
            raise RuntimeError("No VectorDB assigned for memory embedding")

        inputs = self.embed_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to(self.embedding_device)

        outputs = self.embed_model(**inputs)
        last_hidden = outputs.last_hidden_state

        mask = inputs['attention_mask'].unsqueeze(-1)
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)

        return pooled.detach().cpu()

    def add_to_memory(self, vectors: torch.Tensor, metadata: Optional[List[Any]] = None):
        if self.memory is None:
            raise RuntimeError("No VectorDB assigned to this model")
        self.memory.add(vectors.detach().cpu().numpy(), metadata)

    def retrieve_from_memory(self, query: torch.Tensor, top_k: int = 5):
        if self.memory is None:
            raise RuntimeError("No VectorDB assigned to this model")
        return self.memory.search(query.detach().cpu().numpy(), top_k=top_k)




class AdapteraModel:
    """
    ⚠️ Deprecated.

    AdapteraModel has been deprecated and will be removed in the next release.

    Use:
        - AdapteraHFModel (for Hugging Face models)
        - AdapteraLMSModel (for LM Studio models)
    """

    def __new__(cls, *args, **kwargs):
        warnings.warn(
            "AdapteraModel is deprecated and will be removed in the next release. "
            "Use AdapteraHFModel or AdapteraLMSModel instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Default fallback
        return AdapteraHFModel(*args, **kwargs)

# =========================================================
# HUGGINGFACE MODEL (TRAINABLE + QUANT + PEFT + MEMORY)
# =========================================================
class AdapteraHFModel(Memory_Helper):

    def __init__(
        self,
        model_id: str,
        peft_adapter: str | None = None,
        quantization: str | None = None,
        torch_dtype: torch.dtype | None = None,
        vector_db: VectorDB | None = None,
    ):
        if not model_id:
            raise ValueError("model_id must be provided")

        super().__init__()

        self.model_id = model_id
        self.peft_adapter = peft_adapter
        self.quantization = quantization

        self.memory: VectorDB | None = vector_db
        self.use_memory = vector_db is not None

        # ---------------------------
        # Tokenizer
        # ---------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # ---------------------------
        # Quantization config
        # ---------------------------
        quant_config = None
        if quantization is not None:
            if not _BNB_AVAILABLE:
                raise RuntimeError("bitsandbytes is not installed but quantization was requested")

            if quantization == "4bit":
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch_dtype or torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            elif quantization == "8bit":
                quant_config = BitsAndBytesConfig(load_in_8bit=True)
            else:
                raise ValueError("quantization must be: None | '4bit' | '8bit'")

        # ---------------------------
        # Model load
        # ---------------------------
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )

        self.base_model = self.model

        if peft_adapter is not None:
            if os.path.exists(peft_adapter) and os.path.isdir(peft_adapter):
                self.model = PeftModel.from_pretrained(self.model, peft_adapter)
            else:
                raise ValueError(f"PEFT adapter path does not exist or is not a directory: {peft_adapter}")

        self.model.eval()

        # ---------------------------
        # Embedding model (memory)
        # ---------------------------
        if self.use_memory:
            self.embedding_device = "cuda" if torch.cuda.is_available() else "cpu"
            self.embed_tokenizer = AutoTokenizer.from_pretrained(
                "sentence-transformers/all-MiniLM-L6-v2"
            )
            self.embed_model = AutoModel.from_pretrained(
                "sentence-transformers/all-MiniLM-L6-v2"
            ).to(self.embedding_device)
            self.embed_model.eval()

    # ---------------------------
    # GENERATE (HF)
    # ---------------------------
    def generate(
        self,
        prompt: str,
        min_new_tokens: int = 16,
        max_new_tokens: int = 128,
        top_p: float = 0.9,
        temperature: float = 0.7,
        do_sample: bool = True,
        top_k_memory: int = 5,
        **kwargs,
    ) -> str:

        full_prompt = prompt

        if self.use_memory and self.memory is not None:
            retrieved = self.retrieve_from_memory(
                self._embed_text(prompt),
                top_k=top_k_memory
            )
            context = "\n".join(str(meta) for _, meta in retrieved if meta is not None)

            if context:
                full_prompt = f"{context}\n\n{prompt}"

        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                min_new_tokens=min_new_tokens,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                top_p=top_p,
                temperature=temperature,
                do_sample=do_sample,
                repetition_penalty=1.1,
                **kwargs,
            )

        return self.tokenizer.decode(
            output_ids[0][input_len:],
            skip_special_tokens=True
        )

# =========================================================
# LM STUDIO MODEL (INFERENCE ONLY, NO ADAPT)
# =========================================================
class AdapteraLMSModel(Memory_Helper):

    def __init__(
        self,
        vector_db: VectorDB = None,
        backend_url: str = "http://127.0.0.1:1234/v1/chat/completions",
        model_name: str = "local-model",
    ):
        super().__init__()

        self.backend_url = backend_url
        self.model_name = model_name

        self.memory: VectorDB | None = vector_db
        self.use_memory = vector_db is not None

        # ---------------------------
        # Embedding model (memory)
        # ---------------------------
        if self.use_memory:
            self.embedding_device = "cuda" if torch.cuda.is_available() else "cpu"
            self.embed_tokenizer = AutoTokenizer.from_pretrained(
                "sentence-transformers/all-MiniLM-L6-v2"
            )
            self.embed_model = AutoModel.from_pretrained(
                "sentence-transformers/all-MiniLM-L6-v2"
            ).to(self.embedding_device)
            self.embed_model.eval()

    # ---------------------------
    # GENERATE (LM STUDIO)
    # ---------------------------
    def generate(self, prompt: str, top_k_memory: int = 5):

        full_prompt = prompt

        if self.use_memory and self.memory is not None:
            retrieved = self.retrieve_from_memory(
                self._embed_text(prompt),
                top_k=top_k_memory
            )
            context = "\n".join(str(meta) for _, meta in retrieved if meta is not None)

            if context:
                full_prompt = f"{context}\n\n{prompt}"

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": full_prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 128,
        }

        response = requests.post(self.backend_url, json=payload)

        if response.status_code != 200:
            raise RuntimeError(f"LM Studio error: {response.text}")

        data = response.json()
        return data["choices"][0]["message"]["content"]