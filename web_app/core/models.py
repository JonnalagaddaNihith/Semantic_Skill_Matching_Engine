from django.db import models
from django.contrib.auth.models import AbstractUser


class User(AbstractUser):
    email = models.EmailField(unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username']

    def __str__(self):
        return self.email


class Prediction(models.Model):
    RELATIONSHIP_CHOICES = [
        ('EXACT', 'Exact Match'),
        ('PARAPHRASE', 'Paraphrase'),
        ('PARTIAL_MATCH', 'Partial Match'),
        ('RELATED', 'Related'),
        ('UNRELATED', 'Unrelated'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='predictions')
    text_a = models.TextField()
    text_b = models.TextField()
    predicted_relationship = models.CharField(max_length=20, choices=RELATIONSHIP_CHOICES)
    confidence_scores = models.JSONField(default=dict)
    task_type = models.CharField(max_length=50, default='skill_matching')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['user', '-created_at']),
            models.Index(fields=['predicted_relationship']),
        ]

    def __str__(self):
        return f"{self.user.email} - {self.predicted_relationship}"


class Bookmark(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='bookmarks')
    prediction = models.ForeignKey(Prediction, on_delete=models.CASCADE, related_name='bookmarks')
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ['user', 'prediction']
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.user.email} - {self.prediction.id}"
