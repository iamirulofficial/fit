import csv
import cv2
import numpy as np
import os
import sys
import tqdm
import streamlit as st
from tempfile import NamedTemporaryFile
import tempfile
import io 
import ffmpeg
import mediapipe as mp
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions.pose import PoseLandmark
from mediapipe.python.solutions import pose as mp_pose
from cv2_plt_imshow import cv2_plt_imshow, plt_format
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import base64
import pickle

# from google.cloud import storage
# import gcsfs
# gcs = storage.Client()

def header1(url): ## Real
    st.markdown(f'<p style="color:#2C8C42;font-size:48px;border-radius:2%;"><center><strong>{url}</strong></center></p>', unsafe_allow_html=True)
    
def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

# def file_selector():
#     storage_client = storage.Client()
#     bucket_name='pushup_dataset'
#     bucket = storage_client.get_bucket(bucket_name)
#     prefix='videos/'
#     iterator = bucket.list_blobs(delimiter='/', prefix=prefix)
#     response = iterator._get_next_page_response()
#     data=[]
#     for i in response['items']:
#         z='gs://'+bucket_name+'/'+i['name']
#         data.append(z)
#     data=data[1:]
#     return data 

def pushup_count():
    
    # header_html = "<img src='data:image/png;base64,{}' class='img-fluid'>".format(
    # img_to_bytes("IIITBH-WEBPAGE-HEADER.png"))
    # st.markdown(header_html, unsafe_allow_html=True)
    
    final_count = 0
    
    st.title('Pushups Counter')  
    st.header("Your Pushups Partner!")
    uploaded_file = st.file_uploader("Upload Video Files",type=['mp4'])
    temporary_location = False
    path = '-'

    if uploaded_file:
        path = 'test_videos/{name}'.format(name=uploaded_file.name)
        new_path = path
        
        g = io.BytesIO(uploaded_file.read())  ## BytesIO Object
        temporary_location = path
        
        with open(temporary_location, 'wb') as out:  ## Open temporary file as bytes
            out.write(g.read())  ## Read bytes into file
        out.close()
            
            
        # gcs.get_bucket('pushup_dataset').blob('videos/{name1}'.format(name1= uploaded_file.name)).upload_from_filename('/home/jupyter/megha/swimmer_stroke_count/test_videos/{name}'.format(name=uploaded_file.name), content_type='mp4')
        
    # filenames=file_selector()
    # filenames = filenames[::-1]
    # filenames.append("-")
    # filenames = filenames[::-1]
    filename = st.selectbox('Select the file', [path])
        
    if filename != "-":
        df = pd.read_csv("fitness_poses_csvs_out_basic.csv")
        x = df[df.columns[2:101]]
        y = df[df.columns[1]].values
        
        clf = pickle.load(open('clf.pkl', 'rb'))
        
        input_file = 'test_videos/{name}'.format(name=filename.split('/')[-1])
        
        video = cv2.VideoCapture(input_file)

        output_file='test_videos/output_{name}.mp4'.format(name=filename.split('/')[-1].split('.')[0])
        
        c = 1
        frameRate = 5

        frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 20
        out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'MP4V'), fps, (frame_width, frame_height))

        output_df = pd.DataFrame()
            
        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.5) as pose:
            i = 0

            frame_no = []
            result = []
            counter = -1
            prev = " "
            
            while True:

                ret, image = video.read()

                if not ret:
                    break

                if c%frameRate != 0:
                    c = c + 1
                    continue


                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image)

                if not results.pose_landmarks:
                    continue

                annotated_image = image.copy()

                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                pose_landmarks = [[lmk.x, lmk.y, lmk.z] for lmk in results.pose_landmarks.landmark]

                frame_height, frame_width = image.shape[:2]

                pose_landmarks *= np.array([frame_width, frame_height, frame_width])

                pose_landmarks = np.around(pose_landmarks, 5).flatten().astype(np.str).tolist()

                new_positions = pd.DataFrame(pose_landmarks)
                new_positions = new_positions.T
                predict = clf.predict(new_positions)
                result.append(predict[0])

                if predict[0] != prev:
                    prev = predict[0]
                    if i == 1:
                        counter = 0
                    if predict[0]=="up":
                        counter = counter + 1

                image = cv2.putText(image, predict[0] , (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1,(128,0,42) , 2)
                image = cv2.putText(image, "count = {}".format(counter) , (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1,(128,0,42) , 2)

                out.write(image)
                frame_no.append(i)
                print(i)

                i = i+1
                c =c+1

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

            output_df["frame_no"] = frame_no
            output_df["predicted_value"] = result

            video.release()
            out.release()
            cv2.destroyAllWindows()
            
            final_count = counter


        output_file_streamlit='test_videos/output_{name}_streamlit.mp4'.format(name=filename.split('/')[-1].split('.')[0])
            
        os.system('ffmpeg -y -i {} -vcodec libx264 {}'.format(output_file,output_file_streamlit))

        avi_file = open('test_videos/output_{name}_streamlit.mp4'.format(name=filename.split('/')[-1].split('.')[0]), 'rb')
        avi_bytes = avi_file.read()
        st.video(avi_bytes)
        
        header1("Completed {} Pushups".format(final_count))
        header1("Great Job!")
            
if __name__=='__main__':
    pushup_count()
    