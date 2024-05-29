import glob
import streamlit as st
import cv2
import torch
from PIL import Image
# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/yolov5s.pt')

def image_input():
    img_file = None
    img_bytes = st.file_uploader("Upload an image", type=['png', 'jpeg', 'jpg'])
    if img_bytes:
        img_file = "uploaded_image." + img_bytes.name.split('.')[-1]
        with open(img_file, 'wb') as out:
            out.write(img_bytes.read())

    if img_file:
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_file, caption="Uploaded Image")
        with col2:
            img = infer_image(img_file)
            st.image(img, caption="Model prediction")

def video_input():
    vid_file = None
    vid_bytes = st.file_uploader("Upload a video", type=['mp4', 'mpv', 'avi'])
    if vid_bytes:
        vid_file = "uploaded_video." + vid_bytes.name.split('.')[-1]
        with open(vid_file, 'wb') as out:
            out.write(vid_bytes.read())

    if vid_file:
        cap = cv2.VideoCapture(vid_file)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        st.markdown("## Video")
        output = st.empty()

        while True:
            ret, frame = cap.read()
            if not ret:
                st.write("Can't read frame, stream ended? Exiting ....")
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output_img = infer_image(frame)
            output.image(output_img)

        cap.release()

def infer_image(img, size=None):
    result = model(img, size=size) if size else model(img)
    result.render()
    image = Image.fromarray(result.ims[0])
    return image

def main():
    st.title("Object Recognition Dashboard")

    # Input option
    input_option = st.sidebar.radio("Select input type", ['image', 'video'])

    if input_option == 'image':
        image_input()
    else:
        video_input()

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass



