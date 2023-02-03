from django.db import models
from django.utils import timezone


# Create your models here.


class AudioLabels(models.Model):
    id = models.AutoField(primary_key=True)
    filename = models.CharField(max_length=255, blank=False)
    dataset = models.CharField(max_length=255, blank=True)
    updatedate = models.DateTimeField(default=timezone.now)
    labeluser = models.CharField(max_length=255, blank=False, default="System")
    labelusername = models.CharField(max_length=255, blank=False, default="")
    labelregions = models.TextField()
    lowquality = models.BooleanField(blank=False, default=False)
    unclear = models.BooleanField(blank=False, default=False)

class TaskProgress(models.Model):
    id = models.AutoField(primary_key=True)
    progress = models.FloatField(default=0)
    progressname = models.CharField(max_length=255,blank=False,default="")
