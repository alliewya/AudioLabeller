import librosa
import soundfile as sf
import math, time
from pathlib import Path
import os, pickle, datetime
from .models import AudioLabels,TaskProgress
import json
from app import factory

def returnPredictions(classifier, filename):

    regions = []
    filepath = os.path.join("app", "static", "audiofiles",filename)
    audio, sr = librosa.load(filepath,sr=22050)
    framelength = sr // 2
    hop_lenght = sr // 12

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

        # #old
        # features = librosa.feature.mfcc(y=frame, sr=sr, n_mfcc=40)
        # #print(features.shape)
        # features = features.reshape(-1)
        # # print(features.shape)
        # # a = features.reshape(1,-1)
        # # print(a.shape)
        # # #print(features.reshape(1,-1))
        # # print("_------_----_---_-")
        # prediction = classifier.predict(features.reshape(1, -1))
        # #print(prediction)
        
        
        
        prediction = classifier.make_prediction(frame)

        if(prediction[0] == '0'):
            
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
        
    regions2 = []

    #remove silent areas
    # intervals = librosa.effects.split(paddedaudio)
    # print(intervals)
    # for i in intervals:
    #     print(librosa.samples_to_time(i[0]))
    #     print(librosa.samples_to_time(i[1]))
    
    try:
        regions = combine_overlapping_times(regions)
    except:
        print(len(regions))

    for r2 in regions:
        a = librosa.time_to_samples(r2['start'])
        b = librosa.time_to_samples(r2['end'])
        intervals = librosa.effects.split(paddedaudio[a:b])
        #print(intervals)
        for i in intervals:
            regions2.append({"start":librosa.samples_to_time(a + i[0]),"end":librosa.samples_to_time(a + i[1])})

    try:
        print(type(regions2))
        print(regions2)
        regions2 = combine_overlapping_times(regions2)
    except:
        print(len(regions2))
        print("Not r2")
    # print(start_index)
    # print(end_index)
    # print(frame.shape)
    #     print(type(start_index))
    #     for i in intervals:
    #         print(librosa.samples_to_time(i[0]))
    #         print(librosa.samples_to_time(i[1]))
    #         regions2.append(
    #             {"start":librosa.samples_to_time(i[0]),"end":librosa.samples_to_time(i[1])
    #             })
    #print(regions2)
    #time.sleep(2)

    predictiondata = {
        "filename": filename,
        "regions": regions2
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
                file_length = librosa.get_duration(y=audio)
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
                        regions[key] = librosa.time_to_samples(regions[key], sr=sr)
                sorted_not_cough = list(not_coughs)
                for regions in sorted_not_cough:
                    for key in regions:
                        regions[key] = librosa.time_to_samples(regions[key], sr=sr)
                
                #Remove shorter than 0.5 seconds - only for not coughs!
                #sorted_regions_samples = [region for region in sorted_regions_samples if region["end"] - region["start"] >= 0.5 * sr]
                sorted_not_cough = [region for region in sorted_not_cough if region["end"] - region["start"] >= 0.5 * sr]

                for i, region in enumerate(sorted_regions_samples):
                    regionaudio = audio[region["start"]:region["end"]]
                    filename = file[:-4]+str(i)+".wav"
                    path = os.path.join("app", "static", "dataset2", "cough", filename)
                    sf.write(path, regionaudio, sr, subtype='PCM_16' )

                for i, region in enumerate(sorted_not_cough):
                    regionaudio = audio[region["start"]:region["end"]]
                    filename = file[:-4]+str(i)+".wav"
                    path = os.path.join("app", "static", "dataset2", "notcough", filename)
                    sf.write(path, regionaudio, sr, subtype='PCM_16' )

                cough.append(sorted_regions_samples)
                not_cough.append(sorted_not_cough)
                    
    print(cough)
    print(not_cough)
    status = {"Number": len(list1), "Files": list1, }

    cough = factory.Dataset(path1=os.path.join("app", "static", "dataset2", "cough"), load=True, samplerate=22050)
    cough.set_label("0")
    print(str(len(cough.samples))+" Cough Samples")
    print(str(len(cough.labels))+" Cough Labels")
    notcough = factory.Dataset(path1=os.path.join("app", "static", "dataset2", "notcough"),load=True, samplerate=22050)
    notcough.set_label("1")
    print(str(len(notcough.samples))+" Not Cough Samples")
    print(str(len(notcough.labels))+" Not Cough Labels")  

    combineddataset = cough.combine_dataset(notcough)
    combineddataset2 = combineddataset


    # notcoughexternal = factory.Dataset(path1=os.path.join("app", "static", "dataset1", "external"), load=True, samplerate=22050)
    # notcoughexternal.set_label("1")
    # print(str(len(notcoughexternal.samples))+"External Not Cough Samples")
    # print(str(len(notcoughexternal.labels))+"External Not Cough Labels")  



    # combineddataset2 = combineddataset.combine_dataset(notcoughexternal)
    # #combineddataset2 = combineddataset
    combineddataset2.add_labels_to_audiosamps()

    # print(str(len(combineddataset2.samples))+" Combined Samples")
    # print(str(len(combineddataset2.labels))+" Combined Labels")

    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        fname = timestamp + "-ds.pickle"
        with open(os.path.join("app", "static", "datasetpickles", fname ), "wb") as f:
            pickle.dump(combineddataset2, f)
        print("Pickled")
    except:
        print("Pickle failed")
    
    return status


def combine_overlapping_times(entries):
    # Sort the list by the 'start' key
    entries.sort(key=lambda x: x['start'])

    # Initialize the result list and store the first entry
    combined_entries = [entries[0]]

    # Iterate through the sorted entries from the second entry onwards
    for i in range(1, len(entries)):
        current_entry = entries[i]
        previous_entry = combined_entries[-1]

        # Check for overlap between the current entry's 'start' and the previous entry's 'end'
        if current_entry['start'] <= previous_entry['end']:
            # Combine the overlapping entries by updating the 'end' of the previous entry
            previous_entry['end'] = max(previous_entry['end'], current_entry['end'])
        else:
            # If there's no overlap, add the current entry to the combined_entries list
            combined_entries.append(current_entry)

    return combined_entries