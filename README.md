# Project Name
Enhanced Brain Image Analysis and Anomaly Detection Using Advanced Image Processing Techniques.

### Project Description:
The project, titled Medical Image Analysis: Enhanced Brain Image Analysis and Anomaly Detection Using Advanced Image Processing Techniques, focuses on improving the quality and diagnostic capability of grayscale brain images. Using traditional image processing methods, it applies techniques such as contrast enhancement, edge detection, selective sharpening, and segmentation to detect and analyze abnormalities efficiently. This computationally lightweight approach avoids reliance on large datasets or machine learning, making it suitable for real-time and small-scale medical applications.


#### Summary - 
This project enhances and processes grayscale brain images to aid medical diagnostics. Key techniques include Contrast Limited Adaptive Histogram Equalization (CLAHE) for contrast enhancement, Sobel and Laplacian operators for edge detection, and gradient-based anomaly detection for identifying high-intensity regions. The processed images are segmented, and region statistics are computed to extract meaningful insights. The methodology is novel in its sequential combination of these traditional techniques, offering interpretability and computational efficiency, without needing deep learning models or large datasets. This makes the method accessible for practical medical imaging tasks, providing better visualization and aiding clinical evaluations.

#### Course concepts used - 
1.	Sobel Filter
2.	Laplacian Filter

   
#### Additional concepts used -
1.  Gradient-Based Anomaly Detection
2.	Selective Sharpening
3.	Segmentation and Contour Analysis
4.  CLAHE

   
#### Dataset - 
Link and/or Explanation if generated

#### Novelty - 
Unique combination of traditional image processing techniques: contrast enhancement, edge detection, selective sharpening, and anomaly detection.

These techniques are applied in a specific sequence to enhance brain images and detect anomalies.

Individually used in various applications, but this exact order and combination have not been applied for detecting anomalies in brain images

No reliance on deep learning or large datasets, ensuring computational efficiency and interpretability.

Tailored for real-time medical imaging, making it ideal for practical use in clinical settings. 

   
### Contributors:
1)  SreeKaavya Katuri(PES1UG22EC295)
•	Designed and implemented the image processing pipeline.
•	Developed contrast enhancement, sharpening and segmentation.
•	Optimized CLAHE, Sobel, and Laplacian filters
•	Assisted in mask generation
2)  Ankushkumar Yemekar (PES1UG22EC039)
•	Focused on the optimization and fine-tuning of the image enhancement and edge detection techniques
•	Optimized CLAHE, Sobel, and Laplacian filters.
•	Developed gradient-based anomaly detection and sharpening.
•	Assisted in segmentation and morphological operations.

3)  Kamala G(PES1UG22EC115)
•	Handled anomaly detection and contour creation.
•	Assisted in gradient-based anomaly detection
•	Supported segmentation mask generation and result analysis.
•	Contributed to statistical analysis and final reporting.



### Steps:
1. Clone Repository
```git clone https://github.com/Digital-Image-Processing-PES-ECE/project-name.git ```

2. Install Dependencies
```pip install -r requirements.txt```

3. Run the Code
```python main.py (for eg.)```

### Outputs:
* Important intermediate steps
* Final output images 

### References:
1.	Zhang, Y., & Wang, L. (2017). Image Processing and Analysis for Medical Imaging
2.	Gonzalez, R. C., & Woods, R. E. (2018). Digital Image Processing (4th ed.). Pearson.
3.	https://github.com/Dhanya-Abhirami/Brain-Tumour-Segmentation/blob/master/Brain%20Tumour%20Detection%20using%20MRI%20Image.ipynb
4.	OpenCV Documentation
5.	https://pyimagesearch.com
6.	https://www.researchgate.net


   
### Limitations and Future Work:
Limitations:
Fixed thresholds and high noise sensitivity can lead to inaccurate results.
Limited shape analysis and lack of machine learning restricts detection detail and adaptability.

Future Work:
Implement adaptive thresholding and noise filtering to improve segmentation.
Add shape analysis, explore deep learning, and adapt for 3D imaging to enhance accuracy and clinical relevance.
Develop an interface for easier adjustments and validate with clinical data.