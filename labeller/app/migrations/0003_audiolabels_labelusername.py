# Generated by Django 4.1.4 on 2023-01-27 01:37

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0002_audiolabels_labelregions'),
    ]

    operations = [
        migrations.AddField(
            model_name='audiolabels',
            name='labelusername',
            field=models.CharField(default='', max_length=255),
        ),
    ]
