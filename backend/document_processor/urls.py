"""
URL configuration for document_processor app.
"""

from django.urls import path
from . import views

urlpatterns = [
    path('extract_entities/', views.extract_entities, name='extract_entities'),
    path('health/', views.health_check, name='health_check'),
    path('document-types/', views.get_supported_document_types, name='document_types'),
]
