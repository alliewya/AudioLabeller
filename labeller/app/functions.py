import librosa
import soundfile as sf
import math
from pathlib import Path
import os
from .models import AudioLabels,TaskProgress
import json

def returnPredictions(classifier, filename):

    regions = []
    filepath = os.path.join("app", "static", "audiofiles",filename)
    audio, sr = librosa.load(filepath)
    framelength = sr // 2
    hop_lenght = sr // 4

    # Pad up to nearest second
    audioduration = librosa.get_duration(y=audio, sr=sr)
    roundedduration = math.ceil(audioduration)
    paddedaudio = librosa.util.fix_length(audio, size = roundedduration * sr)

    # Create frames
    frames = librosa.util.frame(
        paddedaudio, frame_length=framelength, hop_length=hop_lenght, axis=0)

    # flag for merging overlapping frames
    prevframedetected = False

    # Predict each frame label
    for i, frame in enumerate(frames):
        features = librosa.feature.mfcc(y=frame, sr=sr, n_mfcc=40)
        #print(features.shape)
        features = features.reshape(-1)
        # print(features.shape)
        # a = features.reshape(1,-1)
        # print(a.shape)
        # #print(features.reshape(1,-1))
        # print("_------_----_---_-")
        prediction = classifier.predict(features.reshape(1, -1))
        #print(prediction)

        if(prediction == 0):
            starttime = i * hop_lenght / sr
            endtime = starttime + (framelength / sr)
            if(prevframedetected == True):
                regions[-1]["end"] = endtime
            else:
                regions.append({"start": starttime, "end": endtime})
            #regions.append({"start": starttime, "end": endtime})
            prevframedetected = True
        else:
            prevframedetected = False
        #print(knnmodel.predict(features.reshape(1, -1)))
        # print(frame.shape)

    predictiondata = {
        "filename": filename,
        "regions": regions
    }

    return(predictiondata)


def generate_dataset_file():
    """
    Generates a dataset file by iterating through the audio files in the audiofiles directory and performing the following operations:
    1. Filter the audio labels of a particular file by labeluser = '1'
    2. Load the audio file and its length
    3. Split the audio into cough and non-cough regions by finding the start and end times of the labeled cough regions
    4. Convert the start and end times of both the cough and non-cough regions to samples
    5. Store the number of files and their names in a status dictionary

    Returns:
        status (dict): a dictionary containing the number of files and their names
    """

    audio_files = os.listdir(os.path.join("app", "static", "audiofiles"))
    list1 = []
    cough = []
    not_cough = []
    for file in audio_files:
        a = AudioLabels.objects.filter(filename=file, labeluser='1').first()
        if a:
            list1.append(file)
            #print(a)
            #print(type(json.loads(a.labelregions)))
            if json.loads(a.labelregions):
                audio, sr = librosa.load(os.path.join("app", "static", "audiofiles", file), sr=22050)
                file_length = librosa.get_duration(audio)
                not_coughs = []
                sorted_regions = sorted(json.loads(a.labelregions), key=lambda x: x['start'])
                not_coughs.append({"start": 0, "end": sorted_regions[0]["start"]})
                for i, region in enumerate(sorted_regions):
                    if i == len(sorted_regions) - 1:
                        not_coughs.append({"start": region["end"], "end": file_length})
                    elif region['end'] < sorted_regions[i + 1]["start"]:
                        not_coughs.append({"start": region["end"], "end": sorted_regions[i + 1]["start"]})
                
                sorted_regions_samples = list(sorted_regions)
                for regions in sorted_regions_samples:
                    for key in regions:
                        regions[key] = librosa.time_to_samples(regions[key], sr)
                sorted_not_cough = list(not_coughs)
                for regions in sorted_not_cough:
                    for key in regions:
                        regions[key] = librosa.time_to_samples(regions[key], sr)
                
                #Remove shorter than 0.5 seconds - only for not coughs!
                #sorted_regions_samples = [region for region in sorted_regions_samples if region["end"] - region["start"] >= 0.5 * sr]
                sorted_not_cough = [region for region in sorted_not_cough if region["end"] - region["start"] >= 0.5 * sr]

                for i, region in enumerate(sorted_regions_samples):
                    regionaudio = audio[region["start"]:region["end"]]
                    filename = file[:-4]+str(i)+".wav"
                    path = os.path.join("app", "static", "dataset1", "cough", filename)
                    sf.write(path, regionaudio, sr, subtype='PCM_24' )

                for i, region in enumerate(sorted_not_cough):
                    regionaudio = audio[region["start"]:region["end"]]
                    filename = file[:-4]+str(i)+".wav"
                    path = os.path.join("app", "static", "dataset1", "notcough", filename)
                    sf.write(path, regionaudio, sr, subtype='PCM_24' )

                cough.append(sorted_regions_samples)
                not_cough.append(sorted_not_cough)
                    
    print(cough)
    print(not_cough)
    status = {"Number": len(list1), "Files": list1, }
    return status
