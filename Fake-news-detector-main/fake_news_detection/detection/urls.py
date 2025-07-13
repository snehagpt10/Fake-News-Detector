from django.urls import path
from . import views

urlpatterns = [
    path("", views.input_page, name="input_page"),
    path("check_fake_news/", views.check_fake_news, name="check_fake_news"),
    path("improve_prediction/", views.improve_prediction, name="improve_prediction"),  # New URL pattern
    path('feedback_received/', views.feedback_received, name='feedback_received'),
]
