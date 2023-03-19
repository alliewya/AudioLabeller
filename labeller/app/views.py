import os, time

from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
from .apps import AppConfig

from django.http import HttpResponse, JsonResponse, FileResponse
from django.shortcuts import get_object_or_404, redirect
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.views.decorators.csrf import csrf_exempt
from django.core.paginator import Paginator
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm


import requests
import json
import librosa
import pickle
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import math, time, copy
from datetime import datetime
import numpy as np
from app import functions
from app import factory

from django.utils import timezone
from .models import AudioLabels,TaskProgress


with open(os.path.join(settings.BASE_DIR,"app","models","coughknn5mfcc40.bark"), "rb") as f:
    knnmodel = pickle.load(f)

with open(os.path.join(settings.BASE_DIR,"app","models","knn5mfcc40Coughvid.bark"), "rb") as f:
    knnmodel2 = pickle.load(f)

with open(os.path.join(settings.BASE_DIR,"app","models","knn5mfcc40Coughvid1.bark"), "rb") as f:
    knnmodel3 = pickle.load(f)

with open(os.path.join(settings.BASE_DIR,"app","models","SVMmfcc40Coughvid.bark"), "rb") as f:
    svmmodel1 = pickle.load(f)

with open(os.path.join(settings.BASE_DIR,"app","models","SVMmfcc40CoughvidFrames.bark"), "rb") as f:
    svmmodel2 = pickle.load(f)

with open(os.path.join(settings.BASE_DIR,"app","models","predictor.pickle"), "rb") as f:
    predictor = pickle.load(f)

def index(request):
    context = {'user':request.user}
    return render(request, "home.html", context)

def tutorial(request):
    context = {'user':request.user}
    return render(request, "tutorial.html", context)

def utilities(request):
    backups = os.listdir(os.path.join("app","backups",))
    context = {'user':request.user,'backupfiles':backups}
    return render(request, "utility.html", context) 


def model_config(request):
    datasetpickleslist = os.listdir(os.path.join("app","static","datasetpickles"))
    return render(request, 'blocks/config.html', {'datasetpickleslist':datasetpickleslist})

def jsonbackupdownload(request,backup):
    fpath = os.path.join("app","backups",backup)
    response = FileResponse(open(fpath, 'rb'))
    response['Content-Disposition'] = f'attachment; filename="{backup}"'
    return response


def register(request):
    form = UserCreationForm()
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('login')
    return render(request, 'registration/registration_form.html', {'form': form})


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


def audiowavesnewtouserfromtarget(request, userid, targetuser):
    audiofiles = AudioLabels.objects.filter(labeluser=targetuser).exclude(filename__in=AudioLabels.objects.filter(labeluser=userid).values('filename'))
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

def singlefilewaveuser(request, fname, userid):
    audiofiles = [fname]
    
    audiopredictions = []
    for file in audiofiles:
        try:
            id = int(userid)
            if AudioLabels.objects.filter(filename=file, labeluser =userid).exists():
                labels = AudioLabels.objects.filter(filename=file, labeluser=userid).first()
                audiopredictions.append({'filename':file,'regions':json.loads(labels.labelregions),'labelusername':'','lowquality':labels.lowquality,'unclear':labels.unclear})
        except:
            if AudioLabels.objects.filter(filename=file, labelusername =userid).exists():
                labels = AudioLabels.objects.filter(filename=file, labelusername=userid).first()
                audiopredictions.append({'filename':file,'regions':json.loads(labels.labelregions),'labelusername':'','lowquality':labels.lowquality,'unclear':labels.unclear})
        # else:
        #     pred = functions.returnPredictions(knnmodel, file)
        #     pred['labelusername'] = "New Model Prediction"
        #     audiopredictions.append(pred)
        #     AudioLabels.objects.create(filename=file,labeluser="2",labelusername="Model",labelregions=json.dumps(pred['regions']))
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
            selecteduserid = int(data['userselected'])
            modelperid = [{"id":"2","model":knnmodel,"modelname":"Default Model"},{"id":"3","model":knnmodel2,"modelname":"KNN Model 2"},{"id":"4","model":knnmodel3,"modelname":"KNN Model 3 - Augmented"},{"id":"5","model":svmmodel1,"modelname":"SVM Model 1"},{"id":"6","model":svmmodel2,"modelname":"SVM Model 2 - Augmented"},{"id":"11","model":predictor,"modelname":"Predictor1"}]

            for x in modelperid:
                if int(x["id"]) == selecteduserid:
                    selectedmodel = x["model"]
                    selectedmodelname = x["modelname"]

            print("Reached1")
            audiofiles = os.listdir(os.path.join("app","static","audiofiles"))
            length = len(audiofiles)
            progress = TaskProgress.objects.filter(progressname="PredictDataset").first()
            progress.progress = 0
            progress.save()
            print("Reached2")
            for i, audio in enumerate(audiofiles):
                percentage = (i / length) * 100

                if not AudioLabels.objects.filter(filename=audio, labeluser = selecteduserid).exists():
                    pred = functions.returnPredictions(selectedmodel, audio)
                    AudioLabels.objects.update_or_create(filename=audio,labeluser= selecteduserid, defaults={'updatedate':timezone.now,'labelregions':json.dumps(pred['regions']),'labelusername':selectedmodelname,})   
                elif(data['overwrite']):
                    pred = functions.returnPredictions(selectedmodel, audio)
                    AudioLabels.objects.update_or_create(filename=audio,labeluser= selecteduserid, defaults={'updatedate':timezone.now,'labelregions':json.dumps(pred['regions']),'labelusername':selectedmodelname,})   


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
    request_body = request.body
    a = json.loads(request_body.decode())
    print(a["processname"])
    if(a["processname"]== "PredictDataset"):
        process = copy.copy(TaskProgress.objects.filter(progressname="PredictDataset").first())
        print(process.progress)

    return JsonResponse({'progress': process.progress, 'task': process.progressname})


class DownloadDBView(APIView):
    def get(self, request):
        file_path = os.path.join(settings.BASE_DIR,"db.sqlite3")
        with open(file_path, 'rb') as fh:
            response = HttpResponse(fh.read(), content_type="application/vnd.ms-excel")
            response['Content-Disposition'] = 'inline; filename=' + os.path.basename(file_path)
            return response    

def providelabelsforcopy(request,key):
    if(key != "8634982"):
        return HttpResponse("Fail")
    else:
        data = AudioLabels.objects.all().values()
        for item in data:
            item['updatedate'] = item['updatedate'].astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            item['labelregions'] = json.loads(item['labelregions'])
        data = list(data)
        return JsonResponse(data, safe=False)

def copylabelsfromexternal(request,key,local):

    if(key != "8634982"):
        return HttpResponse("Fail")
    else:
        backupdata = AudioLabels.objects.all().values()
        backupdata = list(backupdata)
        fname = "{}.txt".format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        fpath = os.path.join("app","backups", fname)
        for item in backupdata:
            item['updatedate'] = item['updatedate'].astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            item['labelregions'] = json.loads(item['labelregions'])
        with open(fpath, 'w') as f:
            f.write(json.dumps(backupdata))

        if( local == "no"):
            response = requests.get('http://audio.catbusiness.net/app/api/providelabelsforcopy/8634982')
            if response.status_code == 200:
                json_data = response.json()
                print(json_data[4])
                created = 0
                updated = 0
                for record in json_data:
                    instance, created = AudioLabels.objects.update_or_create(filename= record['filename'], labeluser = record['labeluser'], defaults={'updatedate':datetime.strptime(record['updatedate'], "%Y-%m-%dT%H:%M:%S.%fZ"),'labelregions':json.dumps(record['labelregions']),'labelusername':record['labelusername'],'lowquality':record['lowquality'],'unclear':record['unclear']})
                    if (instance):
                        updated = updated + 1
                    if (created):
                        created = created + 1
                print("Updated {}", updated)
                print("Created {}", created)
                return JsonResponse({"Updated":updated,"Created":created})
            else :
                return JsonResponse({"Status":"Fail"})
        elif (local == "yes"):
            response = requests.get('http://127.0.0.1:8000/app/api/providelabelsforcopy/8634982')
            if response.status_code == 200:
                json_data = response.json()
                print(json_data[4])
                created = 0
                updated = 0
                for record in json_data:
                    instance, created = AudioLabels.objects.update_or_create(filename= record['filename'], labeluser = record['labeluser'], defaults={'updatedate':datetime.strptime(record['updatedate'], "%Y-%m-%dT%H:%M:%S.%fZ"),'labelregions':json.dumps(record['labelregions']),'labelusername':record['labelusername'],'lowquality':record['lowquality'],'unclear':record['unclear']})
                    if (instance):
                        updated = updated + 1
                    if (created):
                        created = created + 1
                print("Updated ", updated)
                print("Created ", created)
                return JsonResponse({"Timestamp":timezone.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),"Updated":updated,"Created":created})
            else :
                return JsonResponse({"Status":"Fail"})

        return JsonResponse(backupdata, safe=False)

def generateDatasetFile(request):
    
    status = functions.generate_dataset_file()
    print(status)
    return JsonResponse({"Bing":"Bong","Count":status["Number"],"Files": status["Files"]}, safe=False)


@csrf_exempt
def resultsCard(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            current_time = str(int(time.time()))
            cardid = current_time[-5:]
            return render(request, 'parts/model-results.html', {'results':data, 'cardid': cardid})
        except:
            return JsonResponse({"Fail":"Fail"})

def config2(request):
    return render(request, 'blocks/config2.html')


@csrf_exempt
def modelfromconfig(request):
    if request.method == 'POST':
        try:
            # Load and Parse Config
            data = json.loads(request.body)
            print(data)
            print(data['features']['mel_spec'])
            print(data['classifier'])


            # Feature Extractors
            featureconfig = data['features']
            a = factory.MelSpectrogramFeatures()
            a.testfeaturekwargs(**featureconfig['mel_spec'])
            print("a")
            print(featureconfig['mfcc'])
            b = factory.MFCCFeatures(**featureconfig['mfcc'])
            print("b")
            #b.test()

            # Load dataset from pickle and extract features
            try:
                path = os.path.join("app", "static", "datasetpickles", data['datasetpickle'] )
                with open(path, 'rb') as f:
                    dataset = pickle.load(f)
                print("Pickleload")
                print("Audio Loaded " + str(len(dataset.samples)))
                print("Labels Loaded " + str(len(dataset.labels)))
                dataset.add_labels_to_audiosamps()
                
                if(bool(data['dataset']['processing']['enable_normalize'])):
                    dataset.normalize()

                if(bool(data['dataset']['processing']['enable_segment'])):
                    dataset.segment()
                if(bool(data['dataset']['processing']['enable_trim'])):
                    dataset.trim_audio()

                if(bool(data['dataset']['fixed']['enable'])):
                    dataset.fix_length(length=data['dataset']['fixed']['fixed_length'])
                    print("fix2")
                elif(bool(data['dataset']['frame']['enable'])):
                    print("frames")
                    dataset.create_frames(frame_size=int(data['dataset']['frame']['frame_frame_size']), hop_length=int(data['dataset']['frame']['frame_hop_length']))
                print(len(dataset.labels))
                print(dataset.labels[3])
                try:
                    print("aa")
                    #mfc,labels = b.features_from_dataset(dataset=dataset)
                    mfc,labels = b.features_from_dataset_multi(dataset=dataset)
                    print("bb")
                    print(data['features'])
                    print(type(data['features']))
                    c = factory.FeaturesFactory(**data['features'])
                    mfc3, labels2 = c.extract_features(dataset=dataset)

                    print("whoop")
                    #print(type(mfc))
                    print(len(mfc))

                    #Classifier
                    try:
                        mdl = None
                        modelfactory = factory.ClassifierFactory(**data['classifier'])
                        if(bool(data['classifier']['knn']['enable'])):
                            mdl = modelfactory.create_classifier()
                            conf = {"Classifier":"Knn","Config":data['classifier']['knn']}
                            print("knn created")
                        elif(bool(data['classifier']['svm']['enable'])):
                            mdl = modelfactory.create_classifier()
                            conf = {"Classifier":"SVM","Config":data['classifier']['svm']}
                        elif(bool(data['classifier']['adaboost']['enable'])):
                            mdl = modelfactory.create_classifier()
                            conf = {"Classifier":"Ada","Config":data['classifier']['adaboost']}
                            print("Adaboos Created")
                        elif(bool(data['classifier']['decision_tree']['enable'])):
                            mdl = modelfactory.create_classifier()
                            conf = {"Classifier":"Decision Tree","Config":data['classifier']['decision_tree']}
                            print("Tree Created")
                        elif(bool(data['classifier']['GMMHMM']['enable'])):
                            mdl = modelfactory.create_classifier()
                            #conf = {"Classifier":"Decision Tree","Config":data['classifier']['decision_tree']}
                            print("GMMHMM Created")
                        print("clf made")
                        print(type(mdl))
                        print(dir(mdl))
                        mfc2 =  np.asarray(mfc)
                        print(mfc2.shape)
                        mfc2 = mfc2.reshape((mfc2.shape[0], -1))
                        print("MFC Reshape")
                        print(mfc2.shape)


                        # Eval
                        try:
                            print("Eval Start")
                            if(bool(data['evaluation']['train_test']['enable'])):
                                print("test train split")
                                params = data['evaluation']
                                if(params['train_test']['tts_random_state']=="None"):
                                    print("train test 1 ")
                                    # X_train, X_test, y_train, y_test = train_test_split(mfc2, labels, 
                                    # test_size=float(params['train_test']['tts_test_size']), 
                                    # shuffle=params['train_test']['tts_shuffle'])
                                    X_train, X_test, y_train, y_test = train_test_split(mfc3, labels2, 
                                    test_size=float(params['train_test']['tts_test_size']), 
                                    shuffle=params['train_test']['tts_shuffle'])
                                    print("train test 2 ")
                                else:
                                    X_train, X_test, y_train, y_test = train_test_split(mfc2, labels, 
                                    test_size=float(params['train_test']['tts_test_size']), 
                                    shuffle=params['train_test']['tts_shuffle'], 
                                    random_state=int(params['train_test']['tts_random_state']))
                                try:
                                    print("mdl fit start")
                                    start_time = time.time()
                                    clf = mdl.fit(X_train, y_train)
                                    end_time = time.time()
                                    timetaken = end_time - start_time
                                    print("mdl fit end")
                                    print(type(clf))
                                    print(dir(clf))
                                except Exception as e:
                                    print("fit fail")
                                    print("An exception occurred:", e)
                                #score = clf.score(X_test, y_test)
                                #print(score)
                                scores = factory.common_scores(clf,X_test=X_test,y_test=y_test)
                                scores["Time"] = timetaken
                                #scores["Classifier"] = conf
                            elif(bool(data['evaluation']['kfold']['enable'])):
                                print("kfold")
                                cv = factory.CrossVal(**data['evaluation'])
                                try:
                                    results = cv.cross_validate(clf=mdl, X=mfc2,y=dataset.labels)
                                    print(results)
                                    scores = {}
                                    scores['test_scores'] = results['test_score'].tolist()
                                    scores['fit_time'] = results['fit_time'].tolist()
                                    scores['score_time'] = results['score_time'].tolist()
                                    print("kfoldbing")
                                except Exception as e:
                                    print("Kfold fail")
                                    print("Exception:",e)
                            elif(bool(data['evaluation']['stratifiedkfold']['enable'])):
                                print("Strat Kfold")
                                cv = factory.CrossVal(**data['evaluation'])
                                try:
                                    results = cv.cross_validate(clf=mdl, X=mfc2,y=dataset.labels)
                                    scores = {}
                                    scores['test_scores'] = results['test_score'].tolist()
                                    scores['fit_time'] = results['fit_time'].tolist()
                                    scores['score_time'] = results['score_time'].tolist()
                                except Exception as e:
                                    print("Strat fail")
                                    print("Exception:",e)
                        except Exception as e:
                            print("Eval Split Fail")
                            print("An exception occurred:", e)
                        
                        predictor = factory.Predictor(featureeex=factory.MFCCFeatures(**featureconfig['mfcc']), clf=clf)
                        with open("predictor.pickle", "wb") as file:
                            pickle.dump(predictor, file)
                        return JsonResponse({"Status": "Success","Scores": scores, "Config": data, "Timestamp": datetime.now().isoformat()})

                        
                    except Exception as e:
                        print("Classifier Fail ")
                        print(data['classifier'])
                        print("An exception occurred:", e)
                except Exception as e:
                    print("Features Fail")
                    print("An exception occurred:", e)
            except Exception as e:
                print("Dataset pickle fail")
                print("An exception occurred:", e)
        except Exception as e:
            print("Fail")
            print("An exception occurred:", e)
    print(request)
    return JsonResponse({'ok':'Fail'})

