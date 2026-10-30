from django.contrib import admin
from .models import User, Prediction, Bookmark

@admin.register(User)
class UserAdmin(admin.ModelAdmin):
    list_display = ['username', 'email', 'created_at']
    search_fields = ['username', 'email']

@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ['user', 'predicted_relationship', 'created_at']
    list_filter = ['predicted_relationship', 'created_at']
    search_fields = ['text_a', 'text_b']

@admin.register(Bookmark)
class BookmarkAdmin(admin.ModelAdmin):
    list_display = ['user', 'prediction', 'created_at']
