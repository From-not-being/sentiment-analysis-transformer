# Sentiment & Reviews Analysis

import pandas as pd
from transformers import pipeline

# 1. Daten-Simulation (Rakuten Ichiba Style)
data = {
    'review_text': [
        "Der Versand war extrem schnell, danke Rakuten!",
        "Das Produkt kam beschädigt an. Sehr enttäuschend.",
        "価格は安かったですが、配送が遅れました。", # Japanisch: Preis war gut, Versand spät.
        "Amazing quality, will buy again."
    ],
    'category': ['Logistics', 'Quality', 'Logistics', 'General']
}
df = pd.DataFrame(data)

# 2. Modell-Setup (Multilinguales Modell für den globalen Markt)
# Wir nutzen ein vortrainiertes BERT-Modell
classifier = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# 3. Analyse
results = classifier(df['review_text'].tolist())

for text, res in zip(df['review_text'], results):
    print(f"Text: {text}\nRating: {res['label']} (Confidence: {res['score']:.2f})\n")
