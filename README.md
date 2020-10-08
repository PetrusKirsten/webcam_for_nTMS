# Webcam for nTMS

## What is a webcam for nTMS?
### Neuronavigation
The neuronavigation is widely used for experiments and techniques of **transcranial magnetic stimulation**, in order to assist in the procedure of **positioning the coil** in the patient's scalp, since, in this way, the variation resulting from the use of different stimulators and individual anatomies are reduced.

### Transcranial magnetic stimulation
**Transcranial magnetic stimulation** (TMS) is a **non-invasive technique of brain stimulation** for therapeutic and diagnostic purposes. It is summarized in a coil capable of generating short and intense magnetic pulses that, when positioned over an individual's head, generate electrical fields in the brain tissue that originate action potentials in the cortical surface neurons.

To assist in positioning the TMS coil, a connection to the neuronavigation is used. Thus, it is called **navigated transcranial magnetic stimulation** (nTMS). This strategy allows real-time monitoring of the TMS coil in relation to neuroimaging.

[InVesalius Navigator](https://github.com/invesalius/invesalius3) [1] is a neuronavigation system developed in Python by the Renato Archer Information Technology Center (Brazil) in partnership with BiomagLab (University of São Paulo, Brazil) used in this project.

![](/Images/invesalius.png)

### The problem in nTMS
A concern of the nTMS operator is the position of the **sensor attached to the patient's head**. There are different types of reference markers for the head; for example: **glasses**, **elastic bands** and **markers** attached directly to the patient's skin. However, all of these are subject to change during the experimental procedures. If this occurs, the **accuracy of neuronavigation is impaired** and the co-registration must be redone, increasing the experimental time.

![](/Images/marker.png)

### The solution: a webcam for nTMS
One way to remedy this problem is to develop an **algorithm with a webcam** capable of **tracking the patient's faces** and the **TMS coil**. In this way, the **head marker becomes the patient's face**. Thus, the use of fixed markers that reduce experimental precision is eliminated.

![](/Images/webcam_tms.png)

## How it works
### Requirements
- Python >=2.7
- [OpenCV](https://opencv.org/)
- [NumPy](https://numpy.org/)
- [SciPy](https://www.scipy.org/)
- [Dlib](http://dlib.net/)
- [imutils](https://github.com/jrosebr1/imutils)

### Head pose estimation
The algorithm uses dlib library for 2D facial detection used. To compute the 3D points with solvePnP, a [anthropometric model](http://aifi.isr.uc.pt/Downloads/OpenGL/glAnthropometric3DModel.cpp) provides a similar 3D facial feature model. The described head pose estimation algorithm in this project is based in [lincolnhard](https://github.com/lincolnhard/head-pose-estimation).

### ArUco markes
To detect the probe and coil, the algorithm uses the [ArUco markers](https://docs.opencv.org/trunk/d5/dae/tutorial_aruco_detection.html). An ArUco marker is a synthetic square marker composed by a wide black border and an inner binary matrix which determines its identifier (id). The dictionary in question is the **DICT_4X4_50**, where the 0 and 1 id correspond the **probe** and 2 and 3 are the **coil**.

![](/Images/coil_and_probe.png)

### Position and orientation filters
In order to increase the accuracy of the system, filters were implemented to compute the position and orientation of the face and ArUco markers.

#### Savitzky-Golay filter
The Savitzky-Golay filter [2] is a finite response filter adapted to smooth a set of data with high frequency noise through the convolution process. The filter fits successive subsets of data points in a window with a low-degree polynomial using the least squares method.

A function based on the Savitzky-Golay filter from the SciPy library was developed, adapted and implemented in the camera's algorithm.

#### Kalman filter
The Kalman filter [3] uses a series of measurements obtained over time that contain inaccuracies and statistical noise. In this way, the filter produces estimates of unknown variables that tend to be more accurate than those that are based on a single measure, resulting in a joint probability distribution over the variables for each period of time.

A function based on the Kalman filter was adapted from [yinguobing](https://github.com/yinguobing/head-pose-estimation/blob/2da5bf229fcf96d5f4fb075a345bd72ff990894f/stabilizer.py) and implemented in the camera's algorithm. The Kalman filter only works in specific cases where the Savitzky-Golay filter is not able to compute. 

![](/Images/gif_head_pose_estimation.gif)
![](/Images/gif_hpe_savitzky-golay.gif)

## References
[1] SOUZA, V. H. et al. Development and characterization of the invesalius navigator software for navigated transcranial magnetic stimulation. Journal of Neuroscience Methods, v. 309, n. 14, p. 109–120, 2018.

[2] SAVITZKY, A.; GOLAY, M. J. Smoothing and differentiation of data by simplified leastsquares procedures.Analytical chemistry, ACS Publications, v. 36, n. 8, p. 1627–1639, 1964

[3] KALMAN, R. A new approach to linear filtering and prediction problems.Journal of Basic Engineering, v. 82, p. 35–45, 1960
