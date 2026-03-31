import pandas as pd
from transformers import pipeline
from tqdm import tqdm

# --- 1. Simulate Product Review Data ---
simulated_data = [
    {"review_comment": "Der Versand war extrem schnell, danke Rakuten!", "date_of_review": "2023-01-15", "rating": 5},
    {"review_comment": "Das Produkt kam beschädigt an. Sehr enttäuschend.", "date_of_review": "2023-01-16", "rating": 1},
    {"review_comment": "価格は安かったですが、配送が遅れました。", "date_of_review": "2023-01-17", "rating": 3}, # Price was cheap, but delivery was delayed.
    {"review_comment": "Amazing quality, will buy again.", "date_of_review": "2023-01-18", "rating": 5},
    {"review_comment": "The customer service was unhelpful and rude.", "date_of_review": "2023-01-19", "rating": 2},
    {"review_comment": "The product broke after a week of use, very poor quality.", "date_of_review": "2023-01-20", "rating": 1},
    {"review_comment": "The website was difficult to navigate and confusing.", "date_of_review": "2023-01-21", "rating": 2},
    {"review_comment": "Highly recommend this item, perfect for my needs!", "date_of_review": "2023-01-22", "rating": 5},
    {"review_comment": "Die Verpackung war aufgerissen, als es ankam.", "date_of_review": "2023-01-23", "rating": 2}, # Packaging was torn when it arrived.
    {"review_comment": "The support team resolved my issue quickly and efficiently.", "date_of_review": "2023-01-24", "rating": 5},
    {"review_comment": "Long delivery times, very frustrating.", "date_of_review": "2023-01-25", "rating": 1},
    {"review_comment": "Good value for money, but the colors are not as expected.", "date_of_review": "2023-01-26", "rating": 3},
    {"review_comment": "The user interface is clunky and outdated.", "date_of_review": "2023-01-27", "rating": 2},
    {"review_comment": "Quality is top-notch, exceeded my expectations.", "date_of_review": "2023-01-28", "rating": 5},
]

df = pd.DataFrame(simulated_data)
print("Simulated Review Data (first 5 rows):")
print(df.head())
print("-" * 50)

# --- 2. Initialize Hugging Face Zero-Shot Classification Pipeline ---
# Using a powerful multi-lingual zero-shot model
# 'device=0' utilizes the GPU for faster processing if available
classifier = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", device=0)

# Define candidate labels for sentiment and categories
sentiment_labels = ['positive', 'negative', 'neutral']
category_labels = ['Logistics', 'Quality', 'Customer Service', 'Website Experience', 'General Product']

def classify_review(comment, classifier, sentiment_labels, category_labels):
    # Classify sentiment
    sentiment_result = classifier(comment, candidate_labels=sentiment_labels, multi_label=False)
    sentiment = sentiment_result['labels'][0]

    # Classify category
    category_result = classifier(comment, candidate_labels=category_labels, multi_label=False)
    category = category_result['labels'][0]

    return sentiment, category

print("Analyzing reviews for sentiment and category...")
tqdm.pandas() # Enable progress_apply for pandas operations
df[['predicted_sentiment', 'category']] = df['review_comment'].progress_apply(
    lambda x: pd.Series(classify_review(x, classifier, sentiment_labels, category_labels))
)
print("Analysis complete.")
print("-" * 50)

# --- 3. Analyze Sentiment by Category and Recommend Improvements ---
print("Sentiment Analysis Results (first 5 rows):")
print(df.head())
print("-" * 50)

print("Aggregated Negative Reviews by Category:")
negative_reviews = df[df['predicted_sentiment'] == 'negative']
negative_counts = negative_reviews['category'].value_counts()
print(negative_counts)
print("-" * 50)

print("\n--- Recommendations for Improvement ---")
if not negative_counts.empty:
    for category, count in negative_counts.items():
        if count > 0: # Focus on categories with negative feedback
            print(f"\nCategory: {category}")
            print(f"  Total negative reviews: {count}")

            # General recommendations based on category
            if category == 'Logistics':
                print("  Recommendation: Improve shipping speed, optimize packaging processes, and provide clearer tracking information to reduce delivery delays and damage.")
            elif category == 'Quality':
                print("  Recommendation: Conduct stricter quality control checks, use more durable materials, and clearly communicate product specifications to manage customer expectations.")
            elif category == 'Customer Service':
                print("  Recommendation: Provide additional training for support staff, reduce response times, and offer multiple channels for customer inquiries to enhance satisfaction.")
            elif category == 'Website Experience':
                print("  Recommendation: Redesign user interface for better navigability, improve search functionality, and optimize for mobile devices to ensure a smooth online experience.")
            elif category == 'General Product':
                print("  Recommendation: Gather more specific feedback on product features and design. Consider A/B testing or user surveys for product enhancements.")
            else:
                print("  Recommendation: Investigate specific issues within this category by analyzing individual negative comments to identify recurring problems.")
else:
    print("No negative reviews found. Keep up the great work!")

print("\n--- Analysis Complete ---")
