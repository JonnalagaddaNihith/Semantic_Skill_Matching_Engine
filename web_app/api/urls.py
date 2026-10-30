from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    PredictionViewSet,
    BookmarkViewSet,
    relationship_types,
    model_metrics,
    CustomAuthToken,
    register,
)

router = DefaultRouter()
router.register(r'predictions', PredictionViewSet, basename='prediction')
router.register(r'bookmarks', BookmarkViewSet, basename='bookmark')

urlpatterns = [
    path('auth/login/', CustomAuthToken.as_view(), name='auth_login'),
    path('auth/register/', register, name='auth_register'),
    path('', include(router.urls)),
    path('relationship-types/', relationship_types, name='relationship_types'),
    path('metrics/', model_metrics, name='model_metrics'),
]
