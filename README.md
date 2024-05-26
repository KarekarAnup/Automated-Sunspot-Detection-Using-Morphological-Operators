# Automated-Sunspot-Detection-Using-Morphological-Operators
This project aimed to develop an automated method for detecting sunspots in H-Alpha images acquired by the spectroheliograph of Coimbra Observatory (OGAUC) and AIA 1600 images captured by the Solar Dynamics Observatory mission (SDO).

# Project Structure

This project is organized into several key sections, each responsible for a specific part of the sunspot detection process. Below is the detailed project structure to be included in the README file:

## Project Structure

### 1. Data
- **data/input/**: Directory containing the input images in TIFF format.
  - `2014oct23_CN_OGAUC.tif`
  - `2014oct23_HMI_SDO.tif`
  - `2014nov29_CN_OGAUC.tif`
  - `2014nov29_HMI_SDO.tif`
- **data/output/**: Directory to store processed images and results.
  - `enhanced_images/`
  - `segmented_images/`
  - `final_results/`

### 2. Notebooks
- **notebooks/**: Directory containing Jupyter notebooks for step-by-step processing and analysis.
  - `1_preprocessing.ipynb`: Notebook for initial image enhancement and noise reduction.
  - `2_edge_detection_and_segmentation.ipynb`: Notebook for edge detection and image segmentation.
  - `3_morphological_operations.ipynb`: Notebook for applying morphological operations.
  - `4_post_processing_and_analysis.ipynb`: Notebook for post-processing and connected components analysis.
  - `5_combined_pipeline.ipynb`: Notebook combining all steps into a complete pipeline.

### 3. Scripts
- **scripts/**: Directory containing Python scripts for each processing step.
  - `preprocess.py`: Script for image pre-processing (blurring, CLAHE).
  - `edge_detection.py`: Script for edge detection (Sobel, Laplacian).
  - `segmentation.py`: Script for image segmentation (thresholding methods).
  - `morphology.py`: Script for morphological operations (dilation, erosion).
  - `post_process.py`: Script for post-processing and analysis (connected components, contour detection).
  - `pipeline.py`: Main script to run the entire pipeline.

### 4. Utils
- **utils/**: Directory containing utility functions used across different scripts and notebooks.
  - `image_io.py`: Functions for reading and writing images.
  - `plotting.py`: Functions for plotting images and results.
  - `filters.py`: Custom filter functions.
  - `morphological_ops.py`: Functions for morphological operations.

### 5. Models (Optional)
- **models/**: Directory for machine learning models (if integrated).
  - `trained_model.pkl`: Pre-trained model for sunspot detection.

### 6. Documentation
- **docs/**: Directory containing documentation files.
  - `README.md`: Project overview, setup instructions, and usage.
  - `CONTRIBUTING.md`: Guidelines for contributing to the project.
  - `references.md`: References and resources used.

### 7. Tests
- **tests/**: Directory containing test cases to validate the functions and scripts.
  - `test_preprocess.py`: Test cases for pre-processing functions.
  - `test_edge_detection.py`: Test cases for edge detection functions.
  - `test_segmentation.py`: Test cases for segmentation functions.
  - `test_morphology.py`: Test cases for morphological operations.
  - `test_post_process.py`: Test cases for post-processing functions.

### Detailed File Descriptions

#### 1. Pre-processing
- `preprocess.py`:
  - **Gaussian Blur**: Applies Gaussian blur to reduce noise.
  - **Median Blur**: Applies median blur to reduce salt-and-pepper noise.
  - **CLAHE**: Enhances contrast using adaptive histogram equalization.

#### 2. Edge Detection and Segmentation
- `edge_detection.py`:
  - **Sobel Filter**: Detects edges using the Sobel operator.
  - **Laplacian Filter**: Detects edges using the Laplacian operator.
  - **Canny Edge Detection**: Detects edges using the Canny algorithm.
- `segmentation.py`:
  - **Otsu's Thresholding**: Segments image using Otsu's method.
  - **Adaptive Thresholding**: Segments image using adaptive thresholding.

#### 3. Morphological Operations
- `morphology.py`:
  - **Dilation**: Expands bright regions in the image.
  - **Erosion**: Shrinks bright regions in the image.
  - **Opening**: Removes small noise using erosion followed by dilation.
  - **Closing**: Fills small holes using dilation followed by erosion.

#### 4. Post-processing and Analysis
- `post_process.py`:
  - **Connected Components Analysis**: Labels and analyzes connected regions.
  - **Contour Detection**: Detects and analyzes contours in the image.

#### 5. Combined Pipeline
- `pipeline.py`: Combines all steps into a complete processing pipeline, from pre-processing to final analysis.

### Setup and Usage

1. **Installation**:
   - Clone the repository.
   - Install required dependencies using `requirements.txt`.

   ```sh
   pip install -r requirements.txt
   ```

2. **Running the Notebooks**:
   - Navigate to the `notebooks/` directory and open the desired notebook in Jupyter.

   ```sh
   jupyter notebook notebooks/1_preprocessing.ipynb
   ```

3. **Running the Scripts**:
   - Execute the individual scripts or the combined pipeline.

   ```sh
   python scripts/preprocess.py
   python scripts/edge_detection.py
   python scripts/segmentation.py
   python scripts/morphology.py
   python scripts/post_process.py
   python scripts/pipeline.py
   ```

4. **Testing**:
   - Run the test cases to validate the functionality of the scripts.

   ```sh
   pytest tests/
   ```

### Conclusion

By following this structured approach, we have developed a robust and scalable pipeline for automated sunspot detection. This project combines various image processing techniques, ensuring accurate and efficient analysis of solar images, which is crucial for space weather research and forecasting.

## Introduction

Sunspots are temporary phenomena on the Sun's photosphere that appear as spots darker than the surrounding areas. These regions are crucial for understanding solar activity, as they are associated with solar flares and other energetic events that can impact space weather and terrestrial communications. The accurate detection and analysis of sunspots are therefore essential for both scientific research and practical applications in space weather forecasting.

Traditionally, sunspot detection has been performed manually by experts, which is a time-consuming and subjective process. With the advent of advanced imaging technologies and the availability of high-resolution solar images from observatories like the Coimbra Observatory (OGAUC) and the Solar Dynamics Observatory (SDO), there is a growing need for automated methods to reliably detect and analyze sunspots.

This project aims to develop an automated algorithm for sunspot detection using a combination of mathematical morphology operators and image processing techniques. By leveraging these methods, we can enhance, segment, and analyze sunspot features in H-Alpha images from OGAUC and AIA 1600 images from SDO. The proposed approach not only improves the accuracy and efficiency of sunspot detection but also lays the groundwork for real-time monitoring and analysis of solar activity.

### Objectives

1. **Enhance Image Quality**: Apply various filtering techniques to reduce noise and improve the contrast of solar images, making sunspots more distinguishable.
2. **Segment Sunspots**: Utilize morphological operations and thresholding methods to accurately segment sunspots from the solar disk.
3. **Edge Detection**: Implement edge detection algorithms to highlight the boundaries of sunspots, aiding in precise segmentation.
4. **Analyze Segmented Regions**: Perform post-processing steps to label and analyze connected components, providing detailed information about sunspot properties such as area and perimeter.
5. **Develop a Robust Pipeline**: Combine multiple image processing techniques to create a comprehensive and reliable pipeline for automated sunspot detection.

### Approach

The algorithm leverages several image processing techniques:

- **Gaussian Blur** and **Median Blur** to reduce noise.
- **CLAHE (Contrast Limited Adaptive Histogram Equalization)** to enhance contrast.
- **Top-Hat and Black-Hat Transformations** to highlight bright and dark features, respectively.
- **Sobel and Laplacian Filters** for edge detection.
- **Adaptive Thresholding** and **Otsu's Thresholding** for segmentation.
- **Morphological Operations** such as dilation and erosion to refine segmented regions.
- **Connected Components Analysis** and **Contour Detection** for detailed analysis of sunspot regions.

### Significance

The successful implementation of this automated sunspot detection algorithm has several significant implications:

- **Efficiency**: Automated detection reduces the time and effort required for manual analysis.
- **Accuracy**: Consistent application of image processing techniques improves detection accuracy and reliability.
- **Real-time Monitoring**: The algorithm can be adapted for real-time processing, facilitating timely monitoring of solar activity.
- **Space Weather Research**: Enhanced detection and analysis of sunspots contribute to a better understanding of solar phenomena and their impact on space weather.

By integrating advanced image processing techniques with morphological operations, this project provides a robust and scalable solution for sunspot detection, paving the way for future research and applications in solar physics and space weather forecasting.

---

## Conclusion

This project aimed to develop an automated method for detecting sunspots in H-Alpha images acquired by the spectroheliograph of Coimbra Observatory (OGAUC) and AIA 1600 images captured by the Solar Dynamics Observatory mission (SDO). The approach utilized various image processing techniques, including morphological operations, edge detection, and segmentation methods, to enhance and identify sunspots accurately.

### Key Steps and Techniques

#### 1. Image Pre-processing
- **Gaussian Blur** and **Median Blur** were used to reduce noise and irrelevant details in the images, which helped in highlighting the significant features such as sunspots.
- **Bilateral Filter** was applied to smooth the image while preserving edges, enhancing the visibility of sunspot boundaries.

#### 2. Image Enhancement
- **Histogram Equalization** and **CLAHE (Contrast Limited Adaptive Histogram Equalization)** were employed to improve the contrast of the images, making sunspots more distinguishable from the solar disk.
- **Top-Hat and Black-Hat Transformations** helped in enhancing bright and dark features, respectively, which facilitated better identification of sunspots against varying backgrounds.

#### 3. Edge Detection and Segmentation
- **Sobel Filter** and **Laplacian Filter** were used for edge detection, highlighting the sharp transitions in intensity that correspond to the edges of sunspots.
- **Canny Edge Detector** provided robust edge detection, ensuring precise identification of sunspot boundaries.
- **Adaptive Thresholding** and **Otsu's Thresholding** were employed to segment the sunspots from the background by determining optimal threshold values for different regions of the image.

#### 4. Morphological Operations
- **Dilation and Erosion** were applied to refine the segmented regions, helping to close gaps and remove small noise elements, thus enhancing the clarity of sunspot boundaries.

#### 5. Post-processing
- **Connected Components Analysis** was used to label and analyze the connected regions in the segmented image, allowing for the identification and filtering of sunspots based on size and shape.
- **Contour Detection** facilitated the precise delineation of sunspot boundaries and enabled further analysis of their properties such as area and perimeter.

### Results

The application of these techniques resulted in the successful detection and segmentation of sunspots in the provided images. The outputs included enhanced images with prominent sunspot features, binary masks highlighting the detected sunspots, and segmented images with clear boundaries.

### Example Outputs

1. **Original Image**:
   - The input H-Alpha and AIA 1600 images showed varying levels of detail and noise.

2. **Pre-processed Image**:
   - Gaussian and Median Blurs reduced noise, while CLAHE improved contrast.
   
3. **Edge Detection**:
   - Sobel and Laplacian filters highlighted the edges of sunspots effectively.
   
4. **Thresholding and Segmentation**:
   - Adaptive and Otsu's Thresholding produced binary masks that segmented sunspots from the solar disk.
   
5. **Morphological Operations**:
   - Refined the segmented regions, enhancing the clarity and accuracy of sunspot detection.

6. **Post-processed Image**:
   - Connected Components Analysis and Contour Detection provided detailed analysis of sunspot regions.

### Conclusion

The combination of these image processing techniques proved to be effective for the automated detection and segmentation of sunspots. By leveraging the strengths of various filters and morphological operators, we achieved a robust and accurate method for identifying sunspots in solar images. This approach can be further refined and adapted to handle different types of solar data, potentially contributing to improved monitoring and analysis of solar activity.

### Future Work

- **Machine Learning Integration**: Incorporate machine learning models to enhance segmentation and classification accuracy.
- **Multi-channel Image Fusion**: Combine data from different wavelengths to improve detection robustness.
- **Automated Feature Extraction**: Develop algorithms for automated feature extraction and analysis of sunspot properties.
- **Real-time Processing**: Optimize the pipeline for real-time sunspot detection and analysis.

By continuing to refine these techniques and integrating advanced methods, we can enhance our understanding and monitoring of solar phenomena, contributing to space weather research and prediction efforts.
