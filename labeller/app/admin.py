from django.contrib import admin
from .models import AudioLabels,TaskProgress,ClassifierResults,TrainingTasks,DatasetPicklesStats
# Register your models here.

admin.site.register(AudioLabels)
admin.site.register(TaskProgress)
admin.site.register(ClassifierResults)
admin.site.register(TrainingTasks)
admin.site.register(DatasetPicklesStats)