from flask import Flask, render_template, request
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)

model = AutoModelForSequenceClassification.from_pretrained("saved_model")
tokenizer = AutoTokenizer.from_pretrained("saved_model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


def predict(text):
    text = text.lower()

    encoding = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    probs = torch.softmax(outputs.logits, dim=1)[0]

    pred = torch.argmax(probs).item()
    confidence = probs[pred].item()

    return pred, confidence, probs.tolist()


def highlight_words(text):
    words = text.split()
    result = []

    for word in words:
        label, _, _ = predict(word)

        if label == 2:
            result.append((word, "hate"))
        elif label == 1:
            result.append((word, "offensive"))
        else:
            result.append((word, "normal"))

    return result


@app.route("/", methods=["GET","POST"])
def home():
    result = None
    probs = None
    highlighted_text = []

    if request.method == "POST":
        text = request.form["text"]

        label, conf, probs = predict(text)
        highlighted_text = highlight_words(text)

        if label == 0:
            result = f"Normal ({conf:.2f})"
        elif label == 1:
            result = f"Offensive ({conf:.2f})"
        else:
            result = f"Hate Speech ({conf:.2f})"

    return render_template(
        "index.html",
        result=result,
        probs=probs,
        highlighted_text=highlighted_text
    )


if __name__ == "__main__":
    app.run(debug=True)