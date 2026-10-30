from rest_framework import viewsets, status
from rest_framework.decorators import api_view, permission_classes, action
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.authtoken.views import ObtainAuthToken
from rest_framework.authtoken.models import Token
from django.contrib.auth import get_user_model
from django.conf import settings
from pathlib import Path
import json

from core.models import Prediction, Bookmark
from core.inference import get_inference
from .serializers import (
    UserSerializer,
    UserRegistrationSerializer,
    PredictionSerializer,
    PredictionCreateSerializer,
    BookmarkSerializer,
)

User = get_user_model()


class CustomAuthToken(ObtainAuthToken):
    def post(self, request, *args, **kwargs):
        # Accept either email or username for login
        email = request.data.get('email')
        username = request.data.get('username')

        if email:
            # Try to find user by email
            try:
                user = User.objects.get(email=email)
                request.data['username'] = user.username
            except User.DoesNotExist:
                pass

        serializer = self.serializer_class(data=request.data,
                                           context={'request': request})
        serializer.is_valid(raise_exception=True)
        user = serializer.validated_data['user']
        token, created = Token.objects.get_or_create(user=user)
        return Response({
            'token': token.key,
            'user_id': user.pk,
            'email': user.email,
            'username': user.username
        })


@api_view(['POST'])
@permission_classes([AllowAny])
def register(request):
    serializer = UserRegistrationSerializer(data=request.data)
    if serializer.is_valid():
        user = serializer.save()
        token, created = Token.objects.get_or_create(user=user)
        return Response({
            'token': token.key,
            'user': UserSerializer(user).data
        }, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class PredictionViewSet(viewsets.ModelViewSet):
    serializer_class = PredictionSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return Prediction.objects.filter(user=self.request.user)

    def create(self, request):
        serializer = PredictionCreateSerializer(data=request.data)
        if serializer.is_valid():
            try:
                inference = get_inference()
                result = inference.predict(
                    serializer.validated_data['text_a'],
                    serializer.validated_data['text_b'],
                    serializer.validated_data.get('task_type', 'skill_matching')
                )

                prediction = Prediction.objects.create(
                    user=request.user,
                    text_a=serializer.validated_data['text_a'],
                    text_b=serializer.validated_data['text_b'],
                    predicted_relationship=result['prediction'],
                    confidence_scores=result['probabilities'],
                    task_type=serializer.validated_data.get('task_type', 'skill_matching')
                )

                return Response(PredictionSerializer(prediction).data,
                              status=status.HTTP_201_CREATED)
            except Exception as e:
                return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=False, methods=['get'])
    def history(self, request):
        queryset = self.get_queryset()
        page = self.paginate_queryset(queryset)
        if page is not None:
            serializer = self.get_serializer(page, many=True)
            return self.get_paginated_response(serializer.data)
        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)


class BookmarkViewSet(viewsets.ModelViewSet):
    serializer_class = BookmarkSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return Bookmark.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        prediction = Prediction.objects.get(
            id=self.request.data['prediction_id'],
            user=self.request.user
        )
        serializer.save(user=self.request.user, prediction=prediction)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def relationship_types(request):
    types = [
        {'value': 'EXACT', 'label': 'Exact Match', 'description': 'Identical skills'},
        {'value': 'PARAPHRASE', 'label': 'Paraphrase', 'description': 'Same meaning, different wording'},
        {'value': 'PARTIAL_MATCH', 'label': 'Partial Match', 'description': 'Some overlap but not identical'},
        {'value': 'RELATED', 'label': 'Related', 'description': 'Connected but different skills'},
        {'value': 'UNRELATED', 'label': 'Unrelated', 'description': 'No meaningful relationship'},
    ]
    return Response(types)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def model_metrics(request):
    metrics_path = Path(settings.MODEL_PATH).parent / 'metrics.json'
    if metrics_path.exists():
        with open(metrics_path) as f:
            return Response(json.load(f))
    return Response({'error': 'Metrics not found'}, status=404)

