import os

from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
from .apps import AppConfig

from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.views.decorators.csrf import csrf_exempt
from django.core.paginator import Paginator
from django.contrib.auth.decorators import login_required


import json
import librosa
import pickle
from sklearn.neighbors import KNeighborsClassifier
import math, time, copy
import numpy as np
from app import functions

from django.utils import timezone
from .models import AudioLabels,TaskProgress


with open(os.path.join(settings.BASE_DIR,"app","models","coughknn5mfcc40.bark"), "rb") as f:
    knnmodel = pickle.load(f)


def index(request):
    context = {'user':request.user}
    return render(request, "home.html", context)



def utilities(request):
    context = {'user':request.user}
    return render(request, "utility.html", context) 


def audiowaves(request):
    audiofile = os.path.abspath("\app\audiofiles\sample.wav")
    audiofilename = "sample.wav"
    regions = [
        {"start": 1, "end": 2}
    ]
    audiofile = {
        "filename": audiofilename,
        "regions": regions
    }
    return render(request, "waves.html", {'audiofile': audiofile})


def audiowaves2(request):
    audiofilelist = []
    audiofile = os.path.abspath("\app\audiofiles\sample.wav")
    audiofilename = "sample.wav"
    regions = [
        {"start": 1, "end": 2}
    ]
    regions2 = [
        {"start": 2, "end": 4},
        {"start": 5, "end": 6}
    ]
    audiofile = {
        "filename": audiofilename,
        "regions": regions
    }
    audiofilelist.append(audiofile)
    audiofile2 = {
        "filename": "boys-yard.wav",
        "regions": regions2
    }
    audiofilelist.append(audiofile2)
    return render(request, "waves2.html", {'audiofilelist': audiofilelist})


def audiowaves3(request):
    # with open("app\models\coughknn5mfcc40.bark", "rb") as f:
    #     knnmodel = pickle.load(f)

    audiofilelist = []
    audiofile = os.path.abspath("\app\audiofiles\sample.wav")
    audiofilename = "sample.wav"
    regions = [
        {"start": 1, "end": 2}
    ]
    regions2 = [
        {"start": 2, "end": 4},
        {"start": 5, "end": 6}
    ]
    audiofile = {
        "filename": audiofilename,
        "regions": regions
    }
    audiofilelist.append(audiofile)
    audiofile2 = {
        "filename": "boys-yard.wav",
        "regions": regions2
    }
    audiofilelist.append(audiofile2)

    regions3 = []
    audio, sr = librosa.load("app\static\sample.wav")
    print(sr)
    framelength = sr // 2
    hop_lenght = sr // 4

    # Pad up to nearest second
    audioduration = librosa.get_duration(y=audio, sr=sr)
    roundedduration = math.ceil(audioduration)
    paddedaudio = librosa.util.fix_length(audio, roundedduration * sr)

    frames = librosa.util.frame(
        paddedaudio, frame_length=framelength, hop_length=hop_lenght, axis=0)
    # print(frames.shape)
    # print(np.size(frames, axis=0))
    # frametimes = librosa.frames_to_time( librosa.util.frame(paddedaudio, frame_length=framelength, hop_length=hop_lenght), sr=sr, hop_length=hop_lenght)
    # print(frametimes.shape)

    # print(frames.shape)
    prevframedetected = False

    for i, frame in enumerate(frames):
        # print("Ahh")
        # print(frame)
        features = librosa.feature.mfcc(y=frame, sr=sr, n_mfcc=40)
        # print(features.shape)
        features = features.reshape(-1)
        # print(features.shape)
        prediction = knnmodel.predict(features.reshape(1, -1))
        if(prediction == 1):
            starttime = i * hop_lenght / sr
            endtime = starttime + (framelength / sr)
            if(prevframedetected == True):
                regions3[-1]["end"] = endtime
            else:
                regions3.append({"start": starttime, "end": endtime})
            prevframedetected = True
        else:
            prevframedetected = False
        #print(knnmodel.predict(features.reshape(1, -1)))
        # print(frame.shape)

    audiofile3 = {
        "filename": audiofilename,
        "regions": regions3
    }
    audiofilelist.append(audiofile3)

    return render(request, "waves2.html", {'audiofilelist': audiofilelist})


# def audiowaves4(request):

#     audiofiles = os.listdir("app\static")
#     audiopredictions = []
#     for file in audiofiles:
#         audiopredictions.append(functions.returnPredictions(knnmodel, file))

#     return render(request, "waves2.html", {'audiofilelist': audiopredictions})

def audiowaves4(request):

    audiofiles = os.listdir(os.path.join("app","static","audiofiles"))
    audiofiles = audiofiles[:5]
    audiopredictions = []
    for file in audiofiles:
        if AudioLabels.objects.filter(filename=file, labeluser ='2').exists():
            labels = AudioLabels.objects.filter(filename=file, labeluser='2').first()
            
            audiopredictions.append({'filename':file,'regions':json.loads(labels.labelregions),'labelusername':'Model'})
        else:
            pred = functions.returnPredictions(knnmodel, file)
            audiopredictions.append(pred)
            AudioLabels.objects.create(filename=file,labeluser="2",labelusername="Model",labelregions=json.dumps(pred['regions']))

    return render(request, "waves2.html", {'audiofilelist': audiopredictions})


# def audiowavespaginated(request):
#     audiofiles = os.listdir("app\static")
#     paginator = Paginator(audiofiles, 5)

#     page_number = request.GET.get('page')

#     # get the current page
#     page_obj = paginator.get_page(page_number)
#     print(page_obj)
#     # loop through the items on the current page
#     audiolist = []
#     for item in page_obj:
#         print(item)
#         # process the data
#         audiolist.append(functions.returnPredictions(knnmodel, item))
#         print(audiolist)
#     # do something with the processed data
#     print(page_obj)
#     return render(request, 'waves3.html', {'page_obj': page_obj, 'audiolist': audiolist, 'user': request.user})

def audiowavespaginated(request):
    audiofiles = os.listdir(os.path.join("app","static","audiofiles"))
    paginator = Paginator(audiofiles, 5)
    paginator.limit_pagination_display = 5

    page_number = request.GET.get('page')

    # get the current page
    page_obj = paginator.get_page(page_number)
    print(page_obj)
    # loop through the items on the current page
    audiopredictions = []
    for file in page_obj:
        if AudioLabels.objects.filter(filename=file, labeluser ='2').exists():
            labels = AudioLabels.objects.filter(filename=file, labeluser='2').first()
            
            audiopredictions.append({'filename':file,'regions':json.loads(labels.labelregions),'labelusername':'Model'})
        else:
            pred = functions.returnPredictions(knnmodel, file)
            pred['labelusername'] = "New Model Prediction"
            audiopredictions.append(pred)
            AudioLabels.objects.create(filename=file,labeluser="2",labelusername="Model",labelregions=json.dumps(pred['regions']))
    # do something with the processed data
    print(page_obj)
    return render(request, 'waves3.html', {'page_obj': page_obj, 'audiolist': audiopredictions, 'user': request.user})

def audiowavesnewtouser(request, userid):
    audiofiles = AudioLabels.objects.filter(labeluser='2').exclude(filename__in=AudioLabels.objects.filter(labeluser=userid).values('filename'))
    paginator = Paginator(audiofiles, 5)
    paginator.limit_pagination_display = 5

    page_number = request.GET.get('page')

    # get the current page
    page_obj = paginator.get_page(page_number)
    print(page_obj)
    # loop through the items on the current page
    audiopredictions = []
    for file in page_obj:
        audiopredictions.append({'filename':file.filename,'regions':json.loads(file.labelregions),'labelusername': file.labelusername,'lowquality':file.lowquality,'unclear':file.unclear})
        
    # do something with the processed data
    print(page_obj)
    return render(request, 'waves3.html', {'page_obj': page_obj, 'audiolist': audiopredictions, 'user': request.user})

def audiowavesbyuser(request, userid):
    audiofiles = AudioLabels.objects.filter(labeluser = userid).order_by("-updatedate")
    paginator = Paginator(audiofiles, 10)
    paginator.limit_pagination_display = 5

    page_number = request.GET.get('page')

    # get the current page
    page_obj = paginator.get_page(page_number)
    print(page_obj)
    # loop through the items on the current page
    audiolist = []
    for item in page_obj:
        print(item)
        reg = item.labelregions
        regions = json.loads(item.labelregions)
        audiolist.append({'filename':item.filename,'regions':json.loads(item.labelregions),'labelusername':item.labelusername,'lowquality':item.lowquality,'unclear':item.unclear})
    # do something with the processed data
    print(page_obj)
    return render(request, 'waves3.html', {'page_obj': page_obj, 'audiolist': audiolist, 'user': request.user})

def singlefilewave(request, fname):
    audiofiles = [fname]
    
    audiopredictions = []
    for file in audiofiles:
        if AudioLabels.objects.filter(filename=file, labeluser ='2').exists():
            labels = AudioLabels.objects.filter(filename=file, labeluser='2').first()
            audiopredictions.append({'filename':file,'regions':json.loads(labels.labelregions),'labelusername':'Model','lowquality':labels.lowquality,'unclear':labels.unclear})
        else:
            pred = functions.returnPredictions(knnmodel, file)
            pred['labelusername'] = "New Model Prediction"
            audiopredictions.append(pred)
            AudioLabels.objects.create(filename=file,labeluser="2",labelusername="Model",labelregions=json.dumps(pred['regions']))
    # do something with the processed data
    
    return render(request, 'waves3.html', {'page_obj': audiopredictions, 'audiolist': audiopredictions, 'user': request.user})



def datasetlist(request):
    audiofiles = os.listdir(os.path.join("app","static","audiofiles"))

    tableobjs = []
    for file in audiofiles:
        tablerow = {'id':'','filename':'','labelledby':[],'modelregions':"",'humanregions':[],'variation':''}
        tablerow['filename'] = file
        tablerow['lowquality'] = False
        tablerow['unclear'] = False
        labels = AudioLabels.objects.filter(filename=file)
        
        for label in labels:
            tablerow['id'] = label.id
            tablerow['labelledby'].append(label.labelusername)
            if(label.labelusername == "Model"):
                tablerow['modelregions'] = (len(json.loads(label.labelregions)))
            else:
                tablerow['humanregions'].append(len(json.loads(label.labelregions)))
            if(label.lowquality):
                tablerow['lowquality'] = True
            if(label.unclear):
                tablerow['unclear'] = True
            
        if (bool(tablerow['humanregions'])):
            if tablerow['modelregions'] != "":
                tablerow['variation'] = int(tablerow['modelregions'])- int(max(tablerow['humanregions']))
        tableobjs.append(tablerow)
        tablejson = json.dumps(tableobjs)
        
    
    return render(request, 'datasetlist.html',{'tableobjs':tableobjs,'tablejson':tablejson})


@csrf_exempt
def save_events_json(request, userid = None):
    if request.method == 'POST':

        try:
            data = json.loads(request.body)
            for audio in data:
                AudioLabels.objects.update_or_create(filename=audio['filename'],labeluser=request.user.id, defaults={'updatedate':timezone.now,'labelregions':json.dumps(audio['regions']),'labelusername':request.user.username,'lowquality':audio['lowquality'],'unclear':audio['unclear']})
            return JsonResponse({'status': 'success'})
        except json.JSONDecodeError as e:
            return JsonResponse({'status': 'error', 'message': 'Invalid JSON'})
    return HttpResponse("OK")



@csrf_exempt
def generate_all_model_predictions(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            audiofiles = os.listdir(os.path.join("app","static","audiofiles"))
            length = len(audiofiles)
            progress = TaskProgress.objects.filter(progressname="PredictDataset").first()
            progress.progress = 0
            progress.save()
            for i, audio in enumerate(audiofiles):
                percentage = (i / length) * 100

                if not AudioLabels.objects.filter(filename=audio, labeluser ='2').exists():
                    pred = functions.returnPredictions(knnmodel, audio)
                    AudioLabels.objects.update_or_create(filename=audio,labeluser="2", defaults={'updatedate':timezone.now,'labelregions':json.dumps(pred['regions']),'labelusername':'Model',})   

                if(i%5 == 0):
                    progress.progress = percentage
                    progress.save()

            progress.progress = 100
            progress.save()
                
        except json.JSONDecodeError as e:
            return JsonResponse({'status': 'error', 'message': 'Invalid JSON'})
    return HttpResponse("OK")

@csrf_exempt
def return_progress(request):
    #data = json.loads(request.body)
    #print(request)
    process = copy.copy(TaskProgress.objects.filter(progressname="PredictDataset").first())


    return JsonResponse({'progress': process.progress, 'task': process.progressname})