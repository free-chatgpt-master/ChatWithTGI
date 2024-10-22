from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def chat_with_ncueatingai(
    model_path: str = "ZoneTwelve/NCUEatingAI-0.5B-v1",
    prompt: str = "What's for lunch?",
    system_prompt: str = "You act like a @ZoneTwelve.",
    max_tokens: int = 64,
):
    # Load the model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Prepare the chat messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

    # Apply chat template
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Tokenize inputs
    inputs = tokenizer(input_text, return_tensors="pt")

    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_tokens,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Example usage
if __name__ == "__main__":
    response = chat_with_ncueatingai(
        prompt="What's a healthy meal?",
        system_prompt="You act like a nutrition expert."
    )
    print("Model Response:", response)
