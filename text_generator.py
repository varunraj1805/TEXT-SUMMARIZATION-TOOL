from transformers import pipeline, set_seed

def generate_text(prompt, max_length=100):
    generator = pipeline('text-generation', model='gpt2')
    set_seed(42)
    return generator(prompt, max_length=max_length, num_return_sequences=1)[0]['generated_text']

if __name__ == "__main__":
    prompt = "Artificial Intelligence is transforming the future by"
    generated = generate_text(prompt)
    print("Generated Text:\n", generated)
