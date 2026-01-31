"""
Hugging Face model implementation with PEFT/QLoRA support.
Optional persistent memory via VectorDB for retrieval-augmented prompts.
Uses a small transformer (MiniLM) for automatic embeddings if memory is provided.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from peft import PeftModel
from typing import Any, List, Optional

try:
    from transformers import BitsAndBytesConfig
    _BNB_AVAILABLE = True
except ImportError:
    _BNB_AVAILABLE = False

from adaptera.memory.core import VectorDB


class AdapteraModel:
    """
    Stateful Hugging Face causal language model with optional PEFT adapters
    and optional persistent vector memory.

    This class wraps a causal LM and keeps model state alive across calls,
    supporting quantization, adapters, and long-term memory via VectorDB.

    Args:
        model_name (str):
            Name or local path of the Hugging Face model.

        peft_adapter (str | None, optional):
            Path to a PEFT adapter (LoRA / QLoRA). If None, no adapter is loaded.

        quantization (str | None, optional):
            Quantization method to use. One of: "4bit", "8bit", or None.

        device_map (str, optional):
            Device mapping strategy for model loading.
            Defaults to "auto".

        torch_dtype (torch.dtype | None, optional):
            Torch dtype for model weights (e.g. torch.float16).

        vector_db (VectorDB | None, optional):
            Optional persistent vector database for long-term memory,
            using `adaptera.VectorDB`.
    """


    def __init__(
        self,
        model_name: str,
        peft_adapter: str | None = None,
        quantization: str | None = None,
        device_map: str = "auto",
        torch_dtype: torch.dtype | None = None,
        vector_db: VectorDB | None = None,  # optional memory
    ):
        if not model_name:
            raise ValueError("model_name must be provided")

        self.model_name = model_name
        self.peft_adapter = peft_adapter
        self.quantization = quantization
        self.memory: VectorDB | None = vector_db
        self.use_memory = vector_db is not None

        # --- Tokenizer ---
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # --- Quantization config ---
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
                raise ValueError("quantization must be one of: None, '4bit', '8bit'")

        # --- Base model ---
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map=device_map,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        )

        # --- PEFT adapter ---
        if peft_adapter is not None:
            self.model = PeftModel.from_pretrained(self.model, peft_adapter)

        self.model.eval()

        # --- Initialize embedding model only if memory is provided ---
        if self.use_memory:
            self.embedding_device = "cuda" if torch.cuda.is_available() else "cpu"
            self.embed_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            self.embed_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(self.embedding_device)
            self.embed_model.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
        """
        Generate text from a prompt.

        If memory is provided, retrieves context from VectorDB and prepends it to the prompt.
        """
        full_prompt = prompt

        # --- Retrieve memory if enabled ---
        if self.use_memory and self.memory is not None:
            retrieved = self.retrieve_from_memory(self._embed_text(prompt), top_k=top_k_memory)
            # Flatten and concatenate context (ignore None metadata)
            context = "\n".join(str(meta) for _, meta in retrieved if meta is not None)
            if context:
                full_prompt = f"{context}\n\n{prompt}"

        # --- Tokenize and generate ---
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                min_new_tokens=min_new_tokens,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                top_p = top_p,
                temperature=temperature,
                do_sample=do_sample,
                repetition_penalty=1.1,
                **kwargs,
            )

        generated_ids = output_ids[0][input_len:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)


    # ------------------------------------------------------------------
    # Memory helper methods
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def _embed_text(self, text: str) -> torch.Tensor:
        """Embed text using the small transformer embedding model."""
        if self.memory is None:
            raise RuntimeError("No VectorDB assigned for memory embedding")
        inputs = self.embed_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.embedding_device)
        outputs = self.embed_model(**inputs)
        last_hidden = outputs.last_hidden_state  # (1, seq_len, hidden_size)
        mask = inputs['attention_mask'].unsqueeze(-1)
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)
        return pooled.detach().cpu()  # shape (1, dim)

    #placeholder function
    def add_to_memory(self, vectors: torch.Tensor, metadata: Optional[List[Any]] = None):
        """Add embeddings to the VectorDB."""
        if self.memory is None:
            raise RuntimeError("No VectorDB assigned to this model")
        self.memory.add(vectors.detach().cpu().numpy(), metadata)

    def retrieve_from_memory(self, query: torch.Tensor, top_k: int = 5):
        """Retrieve nearest neighbors from VectorDB."""
        if self.memory is None:
            raise RuntimeError("No VectorDB assigned to this model")
        return self.memory.search(query.detach().cpu().numpy(), top_k=top_k)
