import os
import sys
import numpy as np
import tempfile

import cv2

root_path = os.getcwd()

from st_on_hover_tabs import on_hover_tabs
from streamlit_option_menu import option_menu
import streamlit_shadcn_ui as ui
from streamlit_image_comparison import image_comparison


from ultralytics import YOLO
import streamlit as st
from zeroDCE.lowlight_test import lowlight

sys.path.insert(0,root_path+"/lane")
from lane.line_fit_video import annotate_image


sys.path.insert(0,root_path+"/msbdn")
from msbdn.dehaze import dehaze

sys.path.insert(0,root_path+"/efficientderain")
from efficientderain.derain import derain

sys.path.insert(0,root_path+"/IATenhance")
from IATenhance.img_demo import exposure



st.set_page_config(layout="wide")

# st.header("Custom tab component for on-hover navigation bar")
st.markdown('<style>' + open('style.css').read() + '</style>', unsafe_allow_html=True)

with st.sidebar:
    tabs = on_hover_tabs(tabName=['图像增强', '车道线检测', '信号灯检测'],
                         iconName=['dashboard', 'money', 'economy'], default_choice=0)


def pil_to_cv(pil_image):
    open_cv_image = np.array(pil_image)
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image


if 'display_type' not in st.session_state:
    st.session_state.display_type = 'image'  # 初始状态为'image'

enhance_function = {
    "低光增强": lowlight,
    "雾天增强": dehaze,
    "雨天增强": derain,
    "过曝矫正": exposure
}

@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

if tabs == '图像增强':
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.header('恶劣环境下图像增强')
        selected2 = option_menu(None, ["低光增强", "雨天增强", "雾天增强", '过曝矫正'],
                                icons=['house', 'cloud-upload', "list-task", 'gear'],
                                menu_icon="cast", default_index=0, orientation="horizontal")
        if selected2 == "低光增强":
            st.markdown("""
            - **意义**：在自动驾驶中，低光增强能够改善夜间或暗光环境下的视觉感知。
            - **实现**：本项目通过使用一个基于深度曲线估计的端到端模型`Zero-DCE`实现低光增强
            """)
        elif selected2 == "雨天增强":
            st.markdown("""
            - **意义**：雨天增强有助于减少雨滴对自动驾驶摄像头捕捉的图像的干扰
            - **实现**：本项目通过使用一种高效的图像去雨网络`efficientderain`利用深度学习技术去除图像中的雨滴，改善视觉效果
            """)
        elif selected2 == "雾天增强":
            st.markdown("""
            - **意义**：雾天增强能够清晰视野，减少由雾引起的视觉模糊
            - **实现**：本项目使用`MSBDN-DFF`，采用多尺度卷积网络结合密集特征融合策略，对雾天图像进行深入的清理和细节恢复
            """)
        elif selected2 == "过曝矫正":
            st.markdown("""
            - **意义**：过曝矫正能够调整自动驾驶摄像头捕捉的过曝图像，使得图像的细节得到恢复
            - **实现**：本项目使用一种轻量级的变换器`Illumination-Adaptive-Transformer`，通过适应局部和全局的图像组件，调整过曝和欠曝部分
            """)
    with col2:
        st.header("Setting")
        source = st.selectbox("Select Source", ["Image", "Video", "Webcam"])
        if source == "Image":
            source_img = st.file_uploader(
                label="Choose an image...",
                type=("jpg", "jpeg", "png", 'bmp', 'webp'),
                key='image'
            )
        if source_img:
            # col1.markdown(source_img, unsafe_allow_html=True)
            if st.session_state.display_type == 'image':
                col1.image(source_img, use_column_width=True)
            else:
                from PIL import Image

                image = Image.open(source_img)

                with st.spinner("Running..."):
                    res = enhance_function[selected2](image)
                    with col1:
                        image_comparison(
                            img1=image,
                            img2=res,
                            label1="增强前",
                            label2="增强后",
                            width=800
                        )
                st.session_state.display_type = "image"
        if ui.button(text="开始增强", key="styled_btn_tailwind", className="bg-orange-500 text-white"):
            if not source_img:
                st.error("请选择一张图片！")
            else:
                st.session_state.display_type = "comparison"
                st.rerun()

elif tabs == '车道线检测':
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.header('车道线检测')
        st.markdown("""
        - **意义**：帮助自动驾驶系统准确识别车道的位置和轨迹，从而保持车辆在行驶道上，并确保行车安全
        - **实现**：本项目结合`OpenCV`，通过梯度阈值边缘检测准确识别和表示车道线
        """)
    with col2:
        st.header("Setting")
        source = st.selectbox("Select Source", ["Image", "Video", "Webcam"])
        if source == "Image":
            source_img = st.file_uploader(
                label="Choose an image...",
                type=("jpg", "jpeg", "png", 'bmp', 'webp'),
                key='image'
            )
        elif source == "Video":
            source_video = st.file_uploader(
                label="Choose a video..."
            )
            if source_video:
                col1.video(source_video)

        if ui.button(text="开始检测", key="styled_btn_tailwind", className="bg-orange-500 text-white"):
            if source == "Image":
                if not source_img:
                    st.error("请选择一张图片！")
                else:
                    from PIL import Image
                    image = Image.open(source_img)
                    image = np.asarray(image)
                    with st.spinner("Running..."):
                        res = annotate_image(image)
                        col1.image(res, use_column_width=True)
            elif source == 'Video':
                if not source_video:
                    st.error("请上传一个视频！")
                else:
                    with col1:
                        with st.spinner("Running..."):
                            tfile = tempfile.NamedTemporaryFile()
                            tfile.write(source_video.read())
                            vid_cap = cv2.VideoCapture(tfile.name)
                            st_frame = col1.empty()
                            while vid_cap.isOpened():
                                success, image = vid_cap.read()
                                if success:
                                    res = annotate_image(image)
                                    st_frame.image(res, use_column_width=True, caption='Detected Video', channels="BGR")
                                else:
                                    vid_cap.release()
                                    break



elif tabs == '信号灯检测':
    col1, col2 = st.columns([0.7, 0.3])
    with col1:
        st.header('信号灯检测')
        st.markdown("""
        - **意义**：帮信号灯检测允许车辆识别交通信号的状态，从而做出停车或通过交叉路口等决策
        - **实现**：本项目使用YOLO模型的最新版本`YOLOv9`实现信号灯的检测
        """)
    with col2:
        st.header("Setting")
        source = st.selectbox("Select Source", ["Image", "Video", "Webcam"])
        confidence = float(st.slider("Select Model Confidence", 30, 100, 50)) / 100
        if source == "Image":
            source_img = st.file_uploader(
                label="Choose an image...",
                type=("jpg", "jpeg", "png", 'bmp', 'webp'),
                key='image'
            )
        elif source == "Video":
            source_video = st.file_uploader(
                label="Choose a video..."
            )
            if source_video:
                col1.video(source_video)

        if ui.button(text="开始检测", key="styled_btn_tailwind", className="bg-orange-500 text-white"):
            if source == "Image":
                if not source_img:
                    st.error("请选择一张图片！")
                else:
                    from PIL import Image
                    image = Image.open(source_img)
                    image = np.asarray(image)
                    model = load_model("./models/best.pt")
                    with st.spinner("Running..."):
                        res = model.track(image, conf=confidence)
                        col1.image(res[0].plot(), use_column_width=True)
            elif source == 'Video':
                if not source_video:
                    st.error("请上传一个视频！")
                else:
                    with col1:
                        with st.spinner("Running..."):
                            tfile = tempfile.NamedTemporaryFile()
                            tfile.write(source_video.read())
                            vid_cap = cv2.VideoCapture(tfile.name)
                            st_frame = col1.empty()
                            model = load_model("./models/best.pt")
                            while vid_cap.isOpened():
                                success, image = vid_cap.read()
                                if success:
                                    res = model.track(image, conf=confidence)
                                    st_frame.image(res[0].plot(), use_column_width=True, caption='Detected Video', channels="BGR")
                                else:
                                    vid_cap.release()
                                    break
