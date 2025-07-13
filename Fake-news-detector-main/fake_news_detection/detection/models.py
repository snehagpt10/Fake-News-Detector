# detection/models.py
from django.db import models
import json

class NewsImprovement(models.Model):
    article_text = models.TextField()
    correct_label = models.CharField(max_length=10)
    pie_chart_base64 = models.TextField()
    # token_saliency = models.JSONField()  # Add this field to store token saliency data

    def __str__(self):
        return f"News Improvement for Article: {self.article_text[:50]}"


class Feedback(models.Model):
    article_text = models.TextField(unique=True)
    label = models.CharField(max_length=10)  # 'Real' or 'Fake'
    created_at = models.DateTimeField(auto_now_add=True)


    def __str__(self):
        return f"Feedback for article: {self.article_text[:50]} - Label: {self.label}"
