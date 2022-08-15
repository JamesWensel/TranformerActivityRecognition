#! /usr/bin/python3

import os
import sys
import csv
import cv2
import time 
import psutil
import random
import numpy as np

from output import Output
from output import printProgressBar

# Create output object to store and output data
output = Output(log_level=0)

# Specify the directory containing the UCF101 dataset.
DATASET_DIR = "UCF 101"

# Specify the height and width to which each video frame will be resized in our dataset.
IMAGE_HEIGHT , IMAGE_WIDTH = 224, 224

# Specify the number of frames of a video that will be fed to the model as one sequence.
SEQUENCE_LENGTH = 20

# Directory to output Processed Numpy Array to
OUTPUT_DIR = "Total Dataset"

# Define name for function to get mem info easier
GET_MEM = psutil.Process(os.getpid()).memory_info

# Record start time and memory of file
output.add_data(time.time(), GET_MEM().rss)

# Name of all classes in video directory
all_classes_names = sorted(os.listdir(DATASET_DIR))

# Specify the list containing the names of the classes used for training. Feel free to choose any set of classes.
CLASSES_LIST = all_classes_names
#CLASSES_LIST = random.sample(all_classes_names, 50)
#CLASSES_LIST = ["WalkingWithDog", "TaiChi", "Swing", "HorseRace"]
#CLASSES_LIST = ["PushUps"]

def frames_extraction(video_path):
    '''
    This function will extract the required frames from a video after resizing and normalizing them.
    Args:
        video_path: The path of the video in the disk, whose frames are to be extracted.
    Returns:
        frames_list: A list containing the resized and normalized frames of the video.
    '''

    # Declare a numpy array to store video frames.
    frames_list = []

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
        #normalized_frame = np.divide(resized_frame.astype('float16'), 255)
        
        # Append the normalized frame into the frames list
        #frames_list.append(normalized_frame)
        frames_list.append(resized_frame)
        
    # Release the VideoCapture object.
    video_reader.release()

    # Return the frames list.
    return frames_list


def create_dataset(output):
    '''
    This function will extract the data of the selected classes and create the required dataset.
    Returns:
        features:          A list containing the extracted frames of the videos.
        labels:            A list containing the indexes of the classes associated with the videos.
        video_files_paths: A list containing the paths of the videos in the disk.
    '''
    # Declared Empty Numpy Arrays to store the features, labels and List to store video file path values.
    features = []
    labels = []
    video_files_paths = []
    
    # Iterating through all the classes mentioned in the classes list
    for class_index, class_name in enumerate(CLASSES_LIST):

        # Display the name of the class whose data is being extracted.
        if output.log_level > 1: 
            print('Extracting Data of Class Number {}: {}'.format(class_index + 1, class_name))
        
        # Record start data for class_name
        output.add_data(time.time(), GET_MEM().rss, identifier="  *  " + class_name)

        # Get the list of video files present in the specific class name directory.
        files_list = os.listdir(os.path.join(DATASET_DIR, class_name))
        
        # Iterate through all the files present in the files list.
        for i, file_name in enumerate(files_list):
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
                
                # Print progress of current class
                printProgressBar(i+1, len(files_list), prefix=class_name, suffix='Complete')
                #break;
                
		# Print the current memory usage and timing statistics, and update csv values
        output.add_data(time.time(), GET_MEM().rss, identifier="  *  " + class_name)
        
    # Records information before creating arrays 
    output.add_data(time.time(), GET_MEM().rss, identifier='Array Creation')
    
    if GET_MEM().rss / 1024**3 >= 60:        
        # Create Memmep array to save processed video files. This is done to circumvent an issue with total ram. The system used to run
        # this code has 125 GB of RAM. The data, after processing, can be 80-100 GB. This means the list used to store the frame data cannot
        # be converted to a numpy array as numpy creates a copy in memory first, then copies the data, using more RAM than is available. Instead, 
        # first write frame data to disk as a numpy memmap array, then delete the data stored in ram and reload the memmap into ram. 
        
        # Get initial timing data for writing to disk
        output.add_data(time.time(), GET_MEM().rss, identifier='Write')
        
        # Create np.memmap array with correct shape to match our data. Save number of videos for reload
        num_vids = len(features)
        feat = np.memmap('temp.npy', dtype='uint8', mode='w+', shape=(num_vids, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
        
        # Copy elements of features array to memory, Print to verify array shape
        feat[:] = features[:] 
        print("Memmap shape: {}".format(feat.shape))
        
        # Push data to Disk
        feat.flush()
        
        # Print time used creating memmap array
        output.add_data(time.time(), GET_MEM().rss, identifier='Write')
        
        # Delete all data in RAM and store data from the process
        output.add_data(time.time(), GET_MEM().rss, identifier='Delete')
        del features
        del feat
        output.add_data(time.time(), GET_MEM().rss, identifier='Delete')
        
        # Recording starting data for reloading the data array
        output.add_data(time.time(), GET_MEM().rss, identifier='Reload')
        
        # Load memmap array. Will not immediately load into RAM, but when saved later will save entire memmap array as normal numpy array
        # and will allow for loading entire dataset into RAM later. Print memory data and features shape to confirm loading
        features = np.memmap('temp.npy', dtype='uint8', mode = 'r', shape=(num_vids, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3))
        print("Memmap shape: {}".format(features.shape))
        
        # Add final data entry for reloading 
        output.add_data(time.time(), GET_MEM().rss, identifier='Reload')
    
    else: 
        # Create np array if there is enough memory for it
        features = np.asarray(features, dtype='uint8') 
        

    # Print final memory and timing data
    output.add_data(time.time(), GET_MEM().rss, identifier='Array Creation')
    
    # Return the frames, class index, and video file path.
    return features, labels, video_files_paths

# Add start data for creating the dataset
output.add_data(time.time(), GET_MEM().rss, identifier='Create Dataset')

# Create the dataset.
features, labels, video_files_paths = create_dataset(output)

# Add output data for creating the dataset
output.add_data(time.time(), GET_MEM().rss, identifier='Create Dataset')

# Print summary of processed array sizes 
#f_size = sys.getsizeof(features) / 1024**3
f_size = (features.size * features.itemsize) / 1024**3
l_size = sys.getsizeof(labels) / 1024
p_size = sys.getsizeof(video_files_paths) / 1024

print("Numpy Array Sizes: ")
print("    {}: {:.4f}GB".format("Features", f_size))
print("    {}: {:.4f}KB".format("Labels", l_size))
print("    {}: {:.4f}KB".format("Video File Paths", p_size))
print("    {}: {:.4f}GB\n".format("Total size", f_size + (l_size / 1024**2) + (p_size / 1024**2)))

# Print array shapes
print("{} Shape: {}".format("Features", features.shape))
print("{} Shape: {}\n".format("Labels", (np.shape(labels))))

# Add start data for numpy array saving
output.add_data(time.time(), GET_MEM().rss, identifier='Save')

# Save features, labels, and video_file_paths to OUTPUT_DIR
np.save(os.path.join(OUTPUT_DIR, "features"), features)
np.save(os.path.join(OUTPUT_DIR,"labels"), labels)
np.save(os.path.join(OUTPUT_DIR,"video_file_paths"), video_files_paths)

# If we created a temporary save array to clear space in RAM, delete it
if os.path.exists("temp.npy"):
    os.remove("temp.npy")

# Add end data for numpy array saving
output.add_data(time.time(), GET_MEM().rss, identifier='Save')

# Add final data entry for end of Runtime
output.add_data(time.time(), GET_MEM().rss, final='True', filename='Format_Time.csv', directory=OUTPUT_DIR)