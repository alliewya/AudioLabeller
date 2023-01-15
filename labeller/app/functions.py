import librosa
import math
from pathlib import Path
import os


def returnPredictions(classifier, filename):

    regions = []
    pat1 = "app/static"
    filepath = os.path.join(pat1, filename)
    audio, sr = librosa.load(filepath)
    framelength = sr // 2
    hop_lenght = sr // 4

    # Pad up to nearest second
    audioduration = librosa.get_duration(y=audio, sr=sr)
    roundedduration = math.ceil(audioduration)
    paddedaudio = librosa.util.fix_length(audio, roundedduration * sr)

    # Create frames
    frames = librosa.util.frame(
        paddedaudio, frame_length=framelength, hop_length=hop_lenght, axis=0)

    # flag for merging overlapping frames
    prevframedetected = False

    # Predict each frame label
    for i, frame in enumerate(frames):
        features = librosa.feature.mfcc(y=frame, sr=sr, n_mfcc=40)
        features = features.reshape(-1)
        prediction = classifier.predict(features.reshape(1, -1))

        if(prediction == 1):
            starttime = i * hop_lenght / sr
            endtime = starttime + (framelength / sr)
            if(prevframedetected == True):
                regions[-1]["end"] = endtime
            else:
                regions.append({"start": starttime, "end": endtime})
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
