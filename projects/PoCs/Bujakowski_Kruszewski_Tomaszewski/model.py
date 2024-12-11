import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def config():
    """
    Return the configuration to use for the model.

    Returns:
        BitsAndBytesConfig: The configuration to use for the model.
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )


def load_model_and_tokenizer(
    model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    device_map: str = "cuda",
    token=None,
):
    """
    Load the model and tokenizer for text generation.

    Args:
        model_name (str): The name of the model to load. Defaults to "meta-llama/Meta-Llama-3-8B-Instruct".
        device_map (str): The device to use for the model. Defaults to "cuda".

    Returns:
        tuple: The model and tokenizer to use for text generation.
    """
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=config(),
        device_map=device_map,
        max_length=4096,
        torch_dtype=torch.bfloat16,
        token=token,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return base_model, tokenizer


def settings():
    """
    Return the settings to use for text generation.

    Returns:
        dict: The settings to use for text generation.
    """
    return {
        "max_length": 350,
        "temperature": 0.1,
        "top_p": 0.9,
        "do_sample": True,
        "top_k": 50,
        "repetition_penalty": 1.1,
    }


class Llama3Generator:
    """
    A class to generate text from a prompt using a model and tokenizer.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        device: str = "cuda",
    ):
        """
        Constructor for the Llama3Generator class.

        Args:
            model (AutoModelForCausalLM): The model to use for text generation.
            tokenizer (AutoTokenizer): The tokenizer to use for text generation.
            device (str): The device to use for the model. Defaults to "cuda".
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate_simple(
        self, prompt: str, gen_settings: dict, completion_only: bool = True
    ) -> str:
        """
        Generate text from a prompt using the model and tokenizer.

        Args:
            prompt (str): The prompt to generate text from.
            gen_settings (dict): The settings to use for text generation.
            completion_only (bool): Whether to return only the generated text. Defaults to True.

        Returns:
            str: The generated text.
        """
        input_ids = self.tokenizer.encode(
            prompt, return_tensors="pt", truncation=True, max_length=4096
        ).to(self.device)

        output = self.model.generate(
            input_ids,
            max_length=gen_settings.get("max_length", 100),
            temperature=gen_settings.get("temperature", 0.7),
            top_p=gen_settings.get("top_p", 0.9),
            top_k=gen_settings.get("top_k", 50),
            repetition_penalty=gen_settings.get("repetition_penalty", 1.2),
            do_sample=gen_settings.get("do_sample", True),
        )

        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response[len(prompt) :] if completion_only else response
