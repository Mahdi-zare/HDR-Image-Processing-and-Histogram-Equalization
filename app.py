import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def create_hdr_mertens(images):

    images = [img for img in images]
    
    # Create Mertens object
    mertens = cv2.createMergeMertens()
    
    # Merge images
    hdr_image = mertens.process(images)
    
    # Convert back to 8-bit
    hdr_image = np.clip(hdr_image * 255, 0, 255).astype(np.uint8)
    
    return hdr_image

def main(HDR_, HIST_):
    global clip_limit, tile_grid_size
    st.title("HDR Image Processing and Histogram Equalization")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].imshow(cv2.imread("head.jpg")); ax[0].set_xticks([]); ax[0].set_yticks([])
    ax[1].imshow(cv2.imread("hist.jpg")); ax[1].set_xticks([]); ax[1].set_yticks([])
    st.pyplot(fig)

    st.write("""
    * ##### The `HDR` Mertens algorithm is an exposure fusion technique that merges multiple images with different exposures into a single high dynamic range (HDR) image, enhancing details in both bright and dark regions without requiring radiometric calibration.
    * ##### `Histogram Equalization` is a technique used in image processing to improve the contrast of an image. It works by redistributing the pixel intensity values so that they span the entire available range (e.g., 0 to 255 for 8-bit images). This enhances details in areas with poor contrast by making the histogram more uniform.
    * ##### Trained and Developed by [Mahdi Zare](https://www.linkedin.com/in/mahdizare22/)
    """)

    st.write("Upload multiple exposure images of the same scene if you want to create an HDR image or not just one enough.")

    uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files and len(uploaded_files) > 0:
        images = []
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            image = np.array(image)
            images.append(image)
        
        st.write("Uploaded Images:")
        for i, image in enumerate(images):
            st.image(image, caption=f"Image {i+1}", use_container_width=True)

        if HDR_ and HIST_:
            hdr_image = create_hdr_mertens(images)
            colors = ('r', 'g', 'b')
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
            CLAHE_list = [clahe.apply(hdr_image[:, :, i]) for i in range(3)]
            CLAHE_image = cv2.merge(CLAHE_list)

            st.image(hdr_image, caption="HDR Image", use_container_width=True)
            st.download_button("Download HDR Image", Image.fromarray(hdr_image).tobytes(), "hdr_image.png", "image/png")

            st.image(CLAHE_image, caption="CLAHE Image", use_container_width=True)
            st.download_button("Download CLAHE Image", Image.fromarray(CLAHE_image).tobytes(), "CLAHE_image.png", "image/png")
        
        elif HDR_ and not HIST_:
            hdr_image = create_hdr_mertens(images)
            st.image(hdr_image, caption="HDR Image", use_container_width=True)
            st.download_button("Download HDR Image", Image.fromarray(hdr_image).tobytes(), "hdr_image.png", "image/png")


        elif not HDR_ and HIST_:
            colors = ('r', 'g', 'b')
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
            CLAHE_list = [clahe.apply(images[0][:, :, i]) for i in range(3)]
            CLAHE_image = cv2.merge(CLAHE_list)

            st.image(CLAHE_image, caption="HIST Image", use_container_width=True)
            st.download_button("Download HIST Image", Image.fromarray(CLAHE_image).tobytes(), "HIST_image.png", "image/png")
        else:
            st.write("Please upload at least two images to create an HDR image.")

st.sidebar.title("User Input Features")

HDR = st.sidebar.selectbox("HDR", ["True", "False"]) == "True"
HIST = st.sidebar.selectbox("Histogram Eq", ["True", "False"]) == "True"

if HIST:
    st.sidebar.markdown("### CLAHE Parameters")
    st.sidebar.info(
        "- **Clip Limit:** Controls contrast enhancement. Higher values increase contrast but may introduce noise.\n"
        "- **Tile Grid Size:** Larger values improve performance but may reduce detail accuracy."
    )
    clip_limit = st.sidebar.slider("Clip Limit", 0, 10, 2, 1)
    tile_grid_size = st.sidebar.slider("Tile Grid Size", 1, 15, 8, 1)

if __name__ == "__main__":
    main(HDR_=HDR, HIST_=HIST)

