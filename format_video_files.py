import os
import cv2
import math
import time 
import random
import numpy as np

# Get the names of all classes/categories in UCF101.
all_classes_names = os.listdir('UCF101')

# Specify the height and width to which each video frame will be resized in our dataset.
IMAGE_HEIGHT , IMAGE_WIDTH = 224, 224

# Specify the number of frames of a video that will be fed to the model as one sequence.
SEQUENCE_LENGTH = 20

# Specify the directory containing the UCF101 dataset.
DATASET_DIR = "UCF101"

# Specify the list containing the names of the classes used for training. Feel free to choose any set of classes.
#CLASSES_LIST = all_classes_names
#CLASSES_LIST = ["WalkingWithDog", "TaiChi", "Swing", "HorseRace"]
#CLASSES_LIST = random.sample(all_classes_names, 20)
CLASSES_LIST = np.load("class_list.npy").tolist()
#np.save("class_list",np.array(CLASSES_LIST))

def frames_extraction(video_path):
    '''
    This function will extract the required frames from a video after resizing and normalizing them.
    Args:
        video_path: The path of the video in the disk, whose frames are to be extracted.
    Returns:
        frames_list: A list containing the resized and normalized frames of the video.
    '''

    # Declare a numpy array to store video frames.
    frames_list = np.empty((0, 224, 224, 3), dtype='uint8')

    # Read the Video File using the VideoCapture object.
    video_reader = cv2.VideoCapture(video_path)

    # Get the total number of frames in the video.
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the interval after which frames will be added to the list.
    skip_frames_window = max(int(video_frames_count/SEQUENCE_LENGTH), 1)

    # Iterate through the Video Frames.
    for frame_counter in range(SEQUENCE_LENGTH):

        # Set the current frame position of the video.
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)

        # Reading the frame from the video.
        success, frame = video_reader.read()

        # Check if Video frame is not successfully read then break the loop
        if not success:
            break

        # Resize the Frame to fixed height and width.
        resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))

        # Convert the resized frame from int to float, then normalize by dividing by 255 so each pixel is between 0 
        # and 1, then finally add a dimension so it can be appended to frames_list 
        # normalized_frame = np.divide(resized_frame.astype('float16'), 255)
        normalized_frame = np.expand_dims(resized_frame, axis=0)
        
        # Append the normalized frame into the frames list
        frames_list = np.append(frames_list, normalized_frame, axis=0)

    # Release the VideoCapture object.
    video_reader.release()

    # Return the frames list.
    return frames_list


def create_dataset():
    '''
    This function will extract the data of the selected classes and create the required dataset.
    Returns:
        features:          A list containing the extracted frames of the videos.
        labels:            A list containing the indexes of the classes associated with the videos.
        video_files_paths: A list containing the paths of the videos in the disk.
    '''
    start = time.time()
    # Declared Empty Numpy Arrays to store the features, labels and List to store video file path values.
    features = []
    labels = []
    video_files_paths = []

    # Iterating through all the classes mentioned in the classes list
    for class_index, class_name in enumerate(CLASSES_LIST):

        # Display the name of the class whose data is being extracted.
        print(f'Extracting Data of Class: {class_name}')

        # Get the list of video files present in the specific class name directory.
        files_list = os.listdir(os.path.join(DATASET_DIR, class_name))
        
        # Iterate through all the files present in the files list.
        for file_name in files_list:
        
            # Get the complete video path.
            video_file_path = os.path.join(DATASET_DIR, class_name, file_name)
            
            # Extract the frames of the video file.
            frames = frames_extraction(video_file_path)
            
            
            # Check if the extracted frames are equal to the SEQUENCE_LENGTH specified above.
            # So ignore the vides having frames less than the SEQUENCE_LENGTH.
            if len(frames) == SEQUENCE_LENGTH:
               
                # Append the data to their repective lists.
                features.append(frames)
                labels.append(class_index)
                video_files_paths.append(video_file_path)
    
    print("Processing time: ", time.time() - start)  
    #feat = np.memmap('test.npy', dtype='float16', mode='w+', shape=(len(features), 20, 224, 224, 3))
    #feat[:] = features[:] 
    #feat.flush() 
    features = np.array(features) 
    labels = np.array(labels)
    
    print("Total time: ", time.time() - start)
    # Return the frames, class index, and video file path.
    return features, labels, video_files_paths

# Create the dataset.
features, labels, video_files_paths = create_dataset()

np.save("features",features)
np.save("labels",labels)
np.save("video_file_paths", video_files_paths)

print("{} Shape: {}".format("Features", features.shape))
print("{} Shape: {}\n".format("Labels", labels.shape))