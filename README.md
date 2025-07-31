Saliency-Based Fake News Detection
A Django-based web application that uses a fine-tuned BERT model to detect fake news and visualize saliency — the parts of the text that most influenced the model’s decision.

🚀 Features
Input a news article and check if it's real or fake
Uses mrm8488/bert-tiny-finetuned-fake-news-detection model from Hugging Face
Visualizes saliency of words in the article
Simple and interactive Django-based frontend
🛠️ Technologies Used
Django
Transformers (Hugging Face)
BERT-Tiny fine-tuned model
HTML/CSS/JS
🧪 How to Run
# Clone the repository
git clone https://github.com/your-username/fake-news-detector.git
cd fake-news-detector

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the Django server
python manage.py runserver
