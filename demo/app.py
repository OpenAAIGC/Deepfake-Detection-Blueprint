import streamlit as st
import warnings
import cv2
import dlib
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import numpy as np
import torch
from retinaface.pre_trained_models import get_model
from blueprint.model import create_model, create_cam
from blueprint.preprocess import crop_face, extract_face, extract_frames
from pathlib import Path
import tempfile
import os
import io

warnings.filterwarnings('ignore')
ROOT_DIR = Path(__file__).parent.parent

def aca(img):
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_float = img.astype(np.float32) / 255.0
    channels = np.moveaxis(img_float, -1, 0)
    sorted_idx = np.argsort(channels, axis=0)
    sorted_values = np.take_along_axis(channels, sorted_idx, axis=0)
    L = sorted_values[0]
    M = sorted_values[1]
    U = sorted_values[2]
    eps = 1e-10
    L_U = L / (U + eps)
    L_M = L / (M + eps)
    M_U = M / (U + eps)
    kernel = np.array([[1, 0, 1], [0, -4, 0], [1, 0, 1]], dtype=np.float32)
    L_U_filtered = cv2.filter2D(np.log(L_U + eps), -1, kernel)
    L_M_filtered = cv2.filter2D(np.log(L_M + eps), -1, kernel)
    M_U_filtered = cv2.filter2D(np.log(M_U + eps), -1, kernel)
    residuals = np.abs(L_U_filtered) + np.abs(L_M_filtered) + np.abs(M_U_filtered)
    p1, p99 = np.percentile(residuals[residuals > 0], (1, 99))
    normalized = np.clip((residuals - p1) / (p99 - p1), 0, 1)
    significant = normalized > 0.1
    result = np.zeros((*residuals.shape, 3), dtype=np.float32)
    result[significant, 0] = 255
    intensity = np.expand_dims(normalized, -1)
    result = result * intensity
    return result.astype(np.uint8)

def perform_ela(img, quality=95, scale=15):
    buffer = io.BytesIO()
    if len(img.shape) == 3 and img.shape[2] == 3:
        working_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        working_img = img.copy()
    img_bytes = cv2.imencode('.jpg', working_img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()
    buffer.write(img_bytes)
    buffer.seek(0)
    compressed_img = cv2.imdecode(np.frombuffer(buffer.read(), np.uint8), cv2.IMREAD_COLOR)
    difference = np.abs(working_img.astype(np.float32) - compressed_img.astype(np.float32)) * scale
    difference = np.clip(difference, 0, 255).astype(np.uint8)
    difference_rgb = cv2.cvtColor(difference, cv2.COLOR_BGR2RGB)
    luminance = np.sum(difference_rgb * np.array([0.299, 0.587, 0.114]), axis=2)
    enhanced = np.zeros_like(difference_rgb)
    for i in range(3):
        enhanced[:,:,i] = np.minimum(difference_rgb[:,:,i] * 2, 255)
    mask = luminance < np.mean(luminance) * 0.5
    enhanced[mask] = [0, 0, 0]
    gamma = 1.4
    enhanced = (((enhanced / 255.0) ** (1/gamma)) * 255).astype(np.uint8)
    return difference, enhanced

@st.cache_resource
def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sbcl = create_model(str(ROOT_DIR / "Weights/weights.tar"), device)
    face_detector = get_model("resnet50_2020-07-20", max_size=1024, device=device)
    face_detector.eval()
    cam_sbcl = create_cam(sbcl)
    dlib_face_detector = dlib.get_frontal_face_detector()
    dlib_face_predictor = dlib.shape_predictor(str(ROOT_DIR / "Weights/shape_predictor_81_face_landmarks.dat"))
    return device, sbcl, face_detector, cam_sbcl, dlib_face_detector, dlib_face_predictor

def predict_image(inp, models):
    device, sbcl, face_detector, cam_sbcl = models[:4]
    targets = [ClassifierOutputTarget(1)]
    if inp is None:
        return None, None
    face_list = extract_face(inp, face_detector)
    if len(face_list) == 0:
        return None, None
    try:
        img = torch.tensor(face_list).to(device)
        if device.type == 'cuda':
            img = img.half()
        img = img / 255
        with torch.no_grad():
            pred = sbcl(img).float().softmax(1)[:, 1].cpu().numpy().tolist()[0]
            confidences = {'Real': 1 - pred, 'Fake': pred}
        img.requires_grad = True
        grayscale_cam = cam_sbcl(input_tensor=img, targets=targets, aug_smooth=True)
        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(face_list[0].transpose(1, 2, 0) / 255, grayscale_cam, use_rgb=True)
        return confidences, cam_image
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None

def predict_video(inp, models):
    device, sbcl, face_detector, cam_sbcl = models[:4]
    targets = [ClassifierOutputTarget(1)]
    if inp is None:
        return None, None
    try:
        face_list, idx_list = extract_frames(inp, 10, face_detector)
        if not face_list:
            return None, None
        img = torch.tensor(face_list).to(device)
        if device.type == 'cuda':
            img = img.half()
        img = img / 255
        with torch.no_grad():
            pred = sbcl(img).float().softmax(1)[:, 1]
            pred_list = []
            idx_img = -1
            for i in range(len(pred)):
                if idx_list[i] != idx_img:
                    pred_list.append([])
                    idx_img = idx_list[i]
                pred_list[-1].append(pred[i].item())
            pred_res = np.array([max(p) for p in pred_list])
            pred = float(pred_res.mean())
        most_fake = np.argmax(pred_res)
        img_for_cam = img[most_fake].unsqueeze(0)
        img_for_cam.requires_grad = True
        grayscale_cam = cam_sbcl(input_tensor=img_for_cam, targets=targets, aug_smooth=True)
        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(face_list[most_fake].transpose(1, 2, 0) / 255, grayscale_cam, use_rgb=True)
        return {'Real': 1 - pred, 'Fake': pred}, cam_image
    except Exception as e:
        st.error(f"Error during video prediction: {str(e)}")
        return None, None

def main():
    with st.sidebar:
        st.title("Deepfake Detection")
        tab = st.radio("Select Input Type:", ["Image", "Video"])
        if tab == "Image":
            st.subheader("Analysis Visualization Options")
            show_gradcam = st.checkbox("GradCAM", value=True)
            show_aca = st.checkbox("ACA", value=False)
            show_ela = st.checkbox("ELA", value=False)
            if show_ela:
                quality = st.slider("JPEG Quality", 0, 100, 95)
                scale = st.slider("ELA Scale", 1, 50, 15)

    models = load_models()
    
    if tab == "Image":
        st.header("Image Deepfake Detection")
        num_cols = 1 + sum([show_gradcam, show_aca, show_ela])
        cols = st.columns(num_cols)
        col_idx = 0
        
        with cols[col_idx]:
            st.subheader("Input")
            image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            if image is not None:
                image = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                st.image(image, caption="Input", use_container_width=True)
                
                if st.button("Analyze"):
                    with st.spinner("Processing..."):
                        confidences, cam_image = predict_image(image, models)
                        if show_gradcam:
                            col_idx += 1
                            with cols[col_idx]:
                                st.subheader("GradCAM")
                                if confidences and cam_image is not None:
                                    st.image(cam_image, caption="Model Focus", use_container_width=True)
                                    for label, conf in confidences.items():
                                        st.progress(conf, text=f"{label}: {conf*100:.1f}%")
                                else:
                                    st.warning("No face detected!")
                        if show_aca:
                            col_idx += 1
                            with cols[col_idx]:
                                st.subheader("ACA")
                                color_map = aca(image)
                                st.image(color_map, use_container_width=True)
                        if show_ela:
                            col_idx += 1
                            with cols[col_idx]:
                                st.subheader("ELA")
                                _, ela_map = perform_ela(image, quality=quality, scale=scale)
                                st.image(ela_map, use_container_width=True)
    else:
        st.header("Video Deepfake Detection")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Input")
            video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
            if video is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4', dir='/home/appuser') as tmp_file:
                    tmp_file.write(video.read())
                    video_path = tmp_file.name
                st.video(video)
                if st.button("Analyze"):
                    with st.spinner("Processing..."):
                        try:
                            confidences, cam_image = predict_video(video_path, models)
                            with col2:
                                st.subheader("Results")
                                if confidences and cam_image is not None:
                                    st.image(cam_image, caption="GradCAM", use_container_width=True)
                                    for label, conf in confidences.items():
                                        st.progress(conf, text=f"{label}: {conf*100:.1f}%")
                                else:
                                    st.warning("No faces detected!")
                        finally:
                            if os.path.exists(video_path):
                                os.unlink(video_path)

if __name__ == "__main__":
    main()