from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load trained model
tokenizer = AutoTokenizer.from_pretrained("conversation_model")
model = AutoModelForSeq2SeqLM.from_pretrained("conversation_model")

def generate(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=30,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Conversation model loaded. Type 'exit' to quit.\n")

while True:
    s = input("Enter gesture words: ")
    if s.lower() == "exit":
        break
    print("→", generate(s))
