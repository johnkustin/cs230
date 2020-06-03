import os
import shutil
import glob
import json
import torch
import cv2
from PIL import Image
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from facenet_pytorch import MTCNN

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Running on device: {device}')

# kaggle competitions download deepfake-detection-challenge -f test_videos.zip
# kaggle competitions download deepfake-detection-challenge -f sample_submission.csv

#TRAIN_DIR = '/kaggle/input/deepfake-detection-challenge/train_sample_videos/'
# test videos directory is deepfake-detection-challenge/test_videos/
# so use the kaggle api to download and unzip test videos to a directory
# cs230/deepfake-detection-challenge/test_videos/

TRAIN_DIR = '/home/ubuntu/cs230/deepfake-detection-challenge/test_videos/'
#TMP_DIR = '/kaggle/tmp/'
TMP_DIR = '/home/ubuntu/cs230/tmp/'
ZIP_NAME = 'dfdc_test_faces.zip'
#METADATA_PATH = TRAIN_DIR + 'metadata.json'
# metadatah path is TRAIN_DIR + 'sample_submission.csv'

#METADATA_PATH = TRAIN_DIR + 'sample_submission.csv'

SCALE = 0.25
N_FRAMES = None

class FaceExtractor:
    def __init__(self, detector, n_frames=None, resize=None):
        """
        Parameters:
            n_frames {int} -- Total number of frames to load. These will be evenly spaced
                throughout the video. If not specified (i.e., None), all frames will be loaded.
                (default: {None})
            resize {float} -- Fraction by which to resize frames from original prior to face
                detection. A value less than 1 results in downsampling and a value greater than
                1 result in upsampling. (default: {None})
        """

        self.detector = detector
        self.n_frames = n_frames
        self.resize = resize
    
    def __call__(self, filename, save_dir):
        """Load frames from an MP4 video, detect faces and save the results.

        Parameters:
            filename {str} -- Path to video.
            save_dir {str} -- The directory where results are saved.
        """

        # Create video reader and find length
        v_cap = cv2.VideoCapture(filename)
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Pick 'n_frames' evenly spaced frames to sample
        if self.n_frames is None:
            sample = np.arange(0, v_len)
        else:
            sample = np.linspace(0, v_len - 1, self.n_frames).astype(int)

        # Loop through frames
        for j in range(v_len):
            success = v_cap.grab()
            if j in sample:
                # Load frame
                success, frame = v_cap.retrieve()
                if not success:
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                
                # Resize frame to desired size
                if self.resize is not None:
                    frame = frame.resize([int(d * self.resize) for d in frame.size])

                save_path = os.path.join(save_dir, f'{j}.png')

                self.detector([frame], save_path=save_path)

        v_cap.release()

#with open(METADATA_PATH, 'r') as f:
#    metadata = json.load(f)
"""
train_df = pd.DataFrame(
    [
        #(video_file, metadata[video_file]['label'], metadata[video_file]['split'], metadata[video_file]['original'] if 'original' in metadata[video_file].keys() else '')
        (video_file, metadata[video_file]['label'])
        for video_file in metadata.keys()
    ],
    #columns=['filename', 'label', 'split', 'original']
    columns=['filename', 'label']
)
"""
#train_df.head()    

# Load face detector
face_detector = MTCNN(margin=14, keep_all=True, factor=0.5, device=device).eval()
# Define face extractor
face_extractor = FaceExtractor(detector=face_detector, n_frames=N_FRAMES, resize=SCALE)
# Get the paths of all train videos
all_train_videos = glob.glob(os.path.join(TRAIN_DIR, '*.mp4'))

os.mkdir(TMP_DIR)
with torch.no_grad():
    for path in tqdm(all_train_videos):
        file_name = path.split('/')[-1]

        save_dir = os.path.join(TMP_DIR, file_name.split(".")[0])

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Detect all faces appear in the video and save them.
        face_extractor(path, save_dir)

os.chdir(TMP_DIR)
#train_df.to_csv('metadata.csv', index=False)

#!zip -r -m -q /kaggle/working/$ZIP_NAME *
shutil.make_archive(ZIP_NAME, 'zip', '*')
