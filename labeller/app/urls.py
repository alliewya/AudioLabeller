from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('audio', views.audiowaves, name='audiowaves'),
    path('audio2', views.audiowaves2, name='audiowaves2'),
    path('audio3', views.audiowaves3, name='audiowaves3'),
    path('audio4', views.audiowaves4, name='audiowaves4'),
    path('audiopages', views.audiowavespaginated, name="audiopaged"),
    path('api/json', views.save_events_json, name='jsonsave')
]
