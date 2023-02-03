from django.urls import path


from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('utilities',views.utilities,name='utilities'),
    #path('audio', views.audiowaves, name='audiowaves'),
    #path('audio2', views.audiowaves2, name='audiowaves2'),
    #path('audio3', views.audiowaves3, name='audiowaves3'),
    #path('audio4', views.audiowaves4, name='audiowaves4'),
    path('audiopages', views.audiowavespaginated, name="audiopaged"),
    path('singlewave/<fname>', views.singlefilewave, name="singlefileview"),
    path('<userid>/audiolabelled', views.audiowavesbyuser,name="audiolabelledbyuser"),
    path('<userid>/newaudio', views.audiowavesnewtouser, name='newaudioforuser'),
    path('datasetlist', views.datasetlist, name='datasetlist'),
    path('api/json', views.save_events_json, name='jsonsave'),
    path('<userid>/api/json', views.save_events_json, name='jsonsave2'),
    path('api/runpredictions', views.generate_all_model_predictions, name='runpredictions'),
    path('api/taskprogress', views.return_progress, name='taskprogress'),
    path('api/databasebackup',views.DownloadDBView.as_view(), name="dbbackup"),
    path('api/copylabelsfromexternal/<key>/<local>',views.copylabelsfromexternal, name="copylabels"),
    path('api/providelabelsforcopy/<key>',views.providelabelsforcopy, name="providelabelsforcopy"),
    path('backups/<backup>', views.jsonbackupdownload, name="jsonbackupdownload"),
    path('register/', views.register, name='register'),
]

