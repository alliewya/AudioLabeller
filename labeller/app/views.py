import os

from django.shortcuts import render
from django.http import HttpResponse
from .apps import AppConfig

from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.views.decorators.csrf import csrf_exempt
from django.core.paginator import Paginator


import librosa
import pickle
from sklearn.neighbors import KNeighborsClassifier
import math
import numpy as np
from app import functions

with open("app\models\coughknn5mfcc40.bark", "rb") as f:
    knnmodel = pickle.load(f)


def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")


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


def audiowaves4(request):

    audiofiles = os.listdir("app\static")
    audiopredictions = []
    for file in audiofiles:
        audiopredictions.append(functions.returnPredictions(knnmodel, file))

    return render(request, "waves2.html", {'audiofilelist': audiopredictions})


def audiowavespaginated(request):
    audiofiles = os.listdir("app\static")
    paginator = Paginator(audiofiles, 5)

    page_number = request.GET.get('page')

    # get the current page
    page_obj = paginator.get_page(page_number)
    print(page_obj)
    # loop through the items on the current page
    audiolist = []
    for item in page_obj:
        print(item)
        # process the data
        audiolist.append(functions.returnPredictions(knnmodel, item))
        print(item)
    # do something with the processed data
    print(page_obj)
    return render(request, 'waves3.html', {'page_obj': page_obj, 'audiolist': audiolist})


@csrf_exempt
def save_events_json(request):
    if request.method == 'POST':
        print(request.body)
    return HttpResponse("OK")
