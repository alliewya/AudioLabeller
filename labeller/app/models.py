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

class ClassifierResults(models.Model):
    id = models.AutoField(primary_key=True)
    data = models.TextField(blank=True)
    data_json = models.JSONField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now = True)

class TrainingTasks(models.Model):

    STATUS_CHOICES = [
    ('pending', 'Pending'),
    ('in_progress', 'In Progress'),
    ('completed', 'Completed'),
    ('failed', 'Failed'),
    ]


    id = models.AutoField(primary_key=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now = True)
    complete = models.BooleanField(default=False)
    config = models.TextField(blank=False)
    config_json = models.JSONField(blank=False)

    priority = models.IntegerField(default=0)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    start_time = models.DateTimeField(null=True, blank=True)
    end_time = models.DateTimeField(null=True, blank=True)
    error_message = models.TextField(blank=True)

    result_data = models.JSONField(blank=True)

        # Optional ForeignKey to ClassifierResults
    classifier_result = models.ForeignKey(
        ClassifierResults,
        on_delete=models.SET_NULL,  # You can choose a suitable on_delete behavior
        null=True,  # Allows the field to be optional
        blank=True,  # Allows the field to be empty in forms
        related_name='training_tasks',  # You can customize the related name
    )

class DatasetPicklesStats(models.Model):
    id = models.AutoField(primary_key=True)
    filename = models.TextField(blank=False)
    stats = models.TextField(blank=False)
    stats_json = models.JSONField()