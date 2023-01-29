from django.apps import AppConfig
import pickle
import os
import pathlib
from sklearn.neighbors import KNeighborsClassifier
from labeller.settings import BASE_DIR


class AppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'app'
    modelpath = os.path.join(BASE_DIR, 'app','models','knn3.bark')
    kn = open(modelpath, 'rb')
    classifier = pickle.load(kn)
    kn.close()
