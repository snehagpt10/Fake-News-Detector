import matplotlib
matplotlib.use('Agg')  # Use the non-GUI backend for rendering
import matplotlib.pyplot as plt
from django.shortcuts import render, redirect
from django.http import HttpResponse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import Saliency
import torch
import numpy as np
from io import BytesIO
import base64
from .models import Feedback, NewsImprovement

# Load the model and tokenizer
model_name = "XSY/albert-base-v2-fakenews-discriminator"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

# Function to compute saliency
def compute_saliency(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    input_ids = inputs["input_ids"]  # Keep as Long
    attention_mask = inputs["attention_mask"].float()  # Make sure this is float for gradient calculation

    # Get the model's input embeddings
    input_embeds = model.albert.embeddings(input_ids)  # Get the embeddings
    input_embeds.retain_grad()  # Enable gradient tracking for input embeddings

    # Forward pass through the model
    encoder_outputs = model.albert.encoder(input_embeds, attention_mask=attention_mask)
    sequence_output = encoder_outputs.last_hidden_state
    logits = model.classifier(sequence_output[:, 0, :])

    # Get the predicted label
    prediction = logits.argmax(dim=1).item()

    # Compute gradients with respect to input embeddings
    saliency = Saliency(lambda embeds: model.classifier(model.albert.encoder(embeds, attention_mask=attention_mask).last_hidden_state[:, 0, :]))
    attributions = saliency.attribute(input_embeds, target=prediction).sum(dim=-1).squeeze(0)

    # Convert tokens and saliency scores to a usable format
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    saliency_scores = attributions.detach().cpu().numpy()

    label = "Real" if prediction == 0 else "Fake"
    return label, tokens, saliency_scores

# Function to generate saliency pie chart
def generate_saliency_pie_chart(tokens, saliency_scores):
    sorted_tokens_with_scores = sorted(zip(tokens, saliency_scores), key=lambda x: abs(x[1]), reverse=True)
    top_tokens, top_scores = zip(*sorted_tokens_with_scores[:10])

    plt.figure(figsize=(8, 8))
    plt.pie(top_scores, labels=top_tokens, autopct='%1.1f%%', startangle=90)
    plt.title("Saliency Scores (Top 10 Tokens)")

    img_buf = BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    img_base64 = base64.b64encode(img_buf.getvalue()).decode('utf-8')
    plt.close()
    return img_base64

# Function to generate dummy pie chart
def generate_dummy_pie_chart(label):
    data = [70, 30] if label.lower() == "real" else [30, 70]
    labels = ["Correct", "Incorrect"]

    plt.figure(figsize=(8, 8))
    plt.pie(data, labels=labels, autopct='%1.1f%%', startangle=90, colors=["#4CAF50", "#FFC107"])
    plt.title(f"Dummy Chart for {label}")

    img_buf = BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    img_base64 = base64.b64encode(img_buf.getvalue()).decode('utf-8')
    plt.close()
    return img_base64

# Views
def input_page(request):
    return render(request, "input.html")

def check_fake_news(request):
    if request.method == "POST":
        article_text = request.POST.get("article_text")

        try:
            improvement = NewsImprovement.objects.get(article_text=article_text)
            label = improvement.correct_label
            pie_chart_base64 = improvement.pie_chart_base64
        except NewsImprovement.DoesNotExist:
            label, tokens, saliency_scores = compute_saliency(article_text)
            pie_chart_base64 = generate_saliency_pie_chart(tokens, saliency_scores)

            NewsImprovement.objects.create(
                article_text=article_text,
                correct_label=label,
                pie_chart_base64=pie_chart_base64
            )

        return render(request, 'results.html', {
            'text': article_text,
            'label': label,
            'pie_chart_base64': pie_chart_base64,
        })

    return render(request, 'input.html')

def improve_prediction(request):
    if request.method == "POST":
        article_text = request.POST.get("article_text")
        correct_label = request.POST.get("correct_label")

        if not correct_label:
            return render(request, 'error.html', {'message': 'Label cannot be empty. Please select Fake or Real.'})

        try:
            improvement = NewsImprovement.objects.get(article_text=article_text)
            improvement.correct_label = correct_label
            improvement.save()
        except NewsImprovement.DoesNotExist:
            pie_chart_base64 = generate_dummy_pie_chart(correct_label)
            NewsImprovement.objects.create(
                article_text=article_text,
                correct_label=correct_label,
                pie_chart_base64=pie_chart_base64
            )

        return redirect('feedback_received')

    return HttpResponse("Invalid request method.")

def submit_feedback(request):
    if request.method == "POST":
        article_text = request.POST.get("article_text", "").strip()
        label = request.POST.get("label", "").strip()

        if not article_text or not label:
            return render(request, "error.html", {"message": "Article text and label cannot be empty."})

        feedback, created = Feedback.objects.update_or_create(
            article_text=article_text,
            defaults={'label': label}
        )

        return redirect('feedback_received')

    return render(request, "error.html", {"message": "Invalid request method."})

def feedback_received(request):
    feedback = Feedback.objects.all().order_by('-created_at')
    return render(request, "feedback_received.html", {"feedback": feedback})
