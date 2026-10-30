from rest_framework import serializers
from django.contrib.auth import get_user_model
from core.models import Prediction, Bookmark

User = get_user_model()

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'created_at']
        read_only_fields = ['id', 'created_at']


class UserRegistrationSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, min_length=8)
    password_confirm = serializers.CharField(write_only=True)

    class Meta:
        model = User
        fields = ['username', 'email', 'password', 'password_confirm']

    def validate(self, attrs):
        if attrs['password'] != attrs['password_confirm']:
            raise serializers.ValidationError("Passwords don't match")
        return attrs

    def create(self, validated_data):
        validated_data.pop('password_confirm')
        password = validated_data.pop('password')
        user = User.objects.create_user(**validated_data)
        user.set_password(password)
        user.save()
        return user


class PredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Prediction
        fields = ['id', 'text_a', 'text_b', 'predicted_relationship',
                  'confidence_scores', 'task_type', 'created_at']
        read_only_fields = ['id', 'predicted_relationship', 'confidence_scores', 'created_at']


class PredictionCreateSerializer(serializers.Serializer):
    text_a = serializers.CharField(max_length=1000)
    text_b = serializers.CharField(max_length=1000)
    task_type = serializers.CharField(default='skill_matching')


class BookmarkSerializer(serializers.ModelSerializer):
    prediction = PredictionSerializer(read_only=True)
    prediction_id = serializers.IntegerField(write_only=True)

    class Meta:
        model = Bookmark
        fields = ['id', 'prediction', 'prediction_id', 'notes', 'created_at']
        read_only_fields = ['id', 'created_at']


class TokenSerializer(serializers.Serializer):
    token = serializers.CharField()
    user = UserSerializer()
