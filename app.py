import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io

def create_hdr_mertens(images):
    mertens = cv2.createMergeMertens()
    hdr_image = mertens.process(images)
    return np.clip(hdr_image * 255, 0, 255).astype(np.uint8)

def apply_clahe(image, clip_limit, tile_grid_size):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    CLAHE_list = [clahe.apply(image[:, :, i]) for i in range(3)]
    return cv2.merge(CLAHE_list)

def download_image(image, filename):
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG")
    st.download_button(f"Download {filename}", buf.getvalue(), f"{filename}.png", "image/png")

def main(HDR_, HIST_):
    st.title("HDR Image Processing and Histogram Equalization")

    # Display reference images
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(cv2.imread("head.jpg")); ax[0].axis('off')
    ax[1].imshow(cv2.imread("hist.jpg")); ax[1].axis('off')
    st.pyplot(fig)

    st.markdown("### About Techniques")
    st.write("""
    * ##### The `HDR Mertens` algorithm is an exposure fusion technique that merges multiple images with different exposures into a single high dynamic range (HDR) image, enhancing details in both bright and dark regions without requiring radiometric calibration.
    * ##### `Histogram Equalization` is a technique used in image processing to improve the contrast of an image. It works by redistributing the pixel intensity values so that they span the entire available range (e.g., 0 to 255 for 8-bit images). This enhances details in areas with poor contrast by making the histogram more uniform.
    * ##### Trained and Developed by [Mahdi Zare](https://www.linkedin.com/in/mahdizare22/)
    """)

    uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        images = [np.array(Image.open(file)) for file in uploaded_files]
        st.write("### Uploaded Images")
        for idx, image in enumerate(images):
            st.image(image, caption=f"Image {idx + 1}", use_container_width=True)

        if HDR_ and HIST_:
            hdr_image = create_hdr_mertens(images)
            CLAHE_image = apply_clahe(hdr_image, clip_limit, tile_grid_size)

            st.image(hdr_image, caption="HDR Image", use_container_width=True)
            download_image(hdr_image, "HDR Image")

            st.image(CLAHE_image, caption="CLAHE Image", use_container_width=True)
            download_image(CLAHE_image, "CLAHE Image")

        elif HDR_:
            hdr_image = create_hdr_mertens(images)
            st.image(hdr_image, caption="HDR Image", use_container_width=True)
            download_image(hdr_image, "HDR Image")

        elif HIST_:
            CLAHE_image = apply_clahe(images[0], clip_limit, tile_grid_size)
            st.image(CLAHE_image, caption="CLAHE Image", use_container_width=True)
            download_image(CLAHE_image, "CLAHE Image")

        else:
            st.warning("Please select at least one processing option: HDR or Histogram Equalization.")

st.sidebar.title("User Input Features")
HDR = st.sidebar.selectbox("HDR", ["True", "False"]) == "True"
HIST = st.sidebar.selectbox("Histogram Eq", ["True", "False"]) == "True"

if HIST:
    st.sidebar.markdown("### CLAHE Parameters")
    clip_limit = st.sidebar.slider("Clip Limit", 0, 10, 2, 1)
    tile_grid_size = st.sidebar.slider("Tile Grid Size", 1, 15, 8, 1)

if __name__ == "__main__":
    main(HDR_=HDR, HIST_=HIST)
