# Generated by Django 4.1.4 on 2023-03-19 07:34

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0005_audiolabels_lowquality_audiolabels_unclear'),
    ]

    operations = [
        migrations.CreateModel(
            name='ClassifierResults',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('data', models.TextField()),
                ('updatedate', models.DateTimeField(default=django.utils.timezone.now)),
            ],
        ),
    ]
