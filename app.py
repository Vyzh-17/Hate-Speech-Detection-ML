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

    probs = torch.softmax(outputs.logits, dim=1)

    pred = torch.argmax(probs, dim=1).item()
    conf = probs[0][pred].item()

    return pred, conf, probs.tolist()[0]

@app.route("/", methods=["GET","POST"])
def home():
    result = None
    probs = None

    if request.method == "POST":
        text = request.form["text"]

        label, conf, probs = predict(text)

        if label == 1:
            result = f"Toxic ({conf:.2f})"
        else:
            result = f"Normal ({conf:.2f})"

    return render_template("index.html", result=result, probs=probs)

if __name__ == "__main__":
    app.run(debug=True)