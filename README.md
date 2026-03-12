A Real-time Depth Aided Adaptive Stereo Generation Framework

# RGBD2VR

RGBD2VR is a real-time XR rendering demo that converts RGB-D camera streams into stereoscopic images for Virtual Reality (VR) headsets and 3D displays.

The system captures RGB and depth images from an Orbbec RGB-D camera and processes them using the RGBD2VR depth engine. The algorithm generates a second virtual view based on the depth information and produces a stereoscopic image pair suitable for VR headsets or 3D monitors.

This project demonstrates how RGB-D sensing and computer vision can be combined with VR rendering to create immersive 3D experiences.

![Demo Video](Doc_Img/RGBD.gif)
Fig.1 RGB and Depth Image Pair

![Demo Video](Doc_Img/SBS_V9.gif)
FIg.2 Side-by-Side VR image
---

# Introduction

Extended Reality (XR), including Virtual Reality (VR), Augmented Reality (AR), and Mixed Reality (MR), relies heavily on high-quality 3D content generation. Traditional stereoscopic capture requires two cameras with a fixed baseline close to the human pupillary distance (PD). However, this approach is difficult to adapt to different hardware platforms and viewer PD values.

RGB-D cameras provide an alternative approach by capturing both color and depth information. Using depth data, it is possible to synthesize a second viewpoint from a single RGB image. This allows real-time generation of stereoscopic images suitable for XR devices.

The **RGBD2VR engine** implements this concept by converting RGB-D frames into stereo image pairs for XR displays.

The application is implemented using:

- **C++**
- **OpenCV** for image processing and rendering
- **StereoKit** for VR headset integration

---

# Hardware Requirements

## 1. RGB-D Camera

Supported camera:

- **Orbbec Femto Mega**
- Other Orbbec RGB-D cameras may also work

https://www.orbbec.com/products/

The camera provides synchronized RGB and depth streams used by the RGBD2VR algorithm.

---

## 2. VR Headset

Tested devices:

- **Meta Quest 2**
- **Meta Quest Pro**

https://www.meta.com/

The VR headset displays the generated stereoscopic image.

---

## 3. Windows PC

Recommended minimum configuration:

| Component | Requirement |
|---|---|
| CPU | Intel i5-4590 / AMD Ryzen 5 1500X or better |
| Memory | 8 GB RAM or more |
| OS | Windows 11 64-bit |
| USB | USB-A or USB-C port (USB-C 3.1 recommended) |

---

## Supported GPUs (Oculus Link)

**NVIDIA**

- GTX 1650 Super
- GTX 1660
- GTX 1660 Ti
- RTX 20 / 30 / 40 series

**AMD**

- Radeon 400 / 500 / 5000 series
- Vega series

---

# Software Requirements

The following software must be installed before running the demo.

### Visual C++ Redistributable

https://aka.ms/vs/17/release/vc_redist.x64.exe

---

### Orbbec Camera Driver

https://dl.orbbec3d.com/dist/drivers/win32/astra-win32-driver-4.3.0.20.zip

---

### Oculus Desktop Application

https://www.oculus.com/download_app/?id=1582076955407037

---

### RGBD2VR Application

Download the release package:

https://github.com/OrbbecDeveloper/RGBD2VR/releases

After downloading, unzip the package.  
The folder will contain:

- `RGBD2VR.exe`
- required runtime libraries

---

# Installation

1. Install **Visual C++ Redistributable**.
2. Install the **Orbbec camera driver**.
3. Install the **Oculus Desktop application**.
4. Download and extract the **RGBD2VR release package**.
5. Connect the **Orbbec RGB-D camera** to the PC using the provided USB-C cable.
6. Turn on the **VR headset**.

---

# Usage

1. Open the **Oculus Desktop App** and sign in.
2. Navigate to **Settings → General**.
3. Enable **Unknown Sources**.

4. Connect the VR headset to the PC.

You may see two prompts inside the headset:

**Allow connected devices to access files**

Allow access.

**Enable Quest Link**

This must be enabled for the application to run.

If not enabled initially, it can be manually enabled from Oculus settings.

---

### Run the Application

1. Enter **Oculus Link mode** on the headset.
2. Run the application on the PC:RGBD2VR.exe
3. The RGB-D camera stream will be processed.
4. The generated stereoscopic image will be displayed in the VR headset.

---

# Application Architecture

The RGBD2VR pipeline consists of the following components.
RGB-D Camera
│
│ RGB + Depth Frames
▼
RGBD2VR Algorithm (Depth Engine)
│
│ View synthesis from depth
▼
Stereo Image Pair
│
├── VR Headset (StereoKit)
└── 3D Display (OpenCV rendering)


### RGBD2VR Algorithm

The RGBD2VR algorithm generates a second view image by shifting RGB pixels according to their depth values. This creates a stereoscopic image pair suitable for XR display systems.

Key processing steps:

1. RGB and depth frame acquisition  
2. Depth to RGB registration  
3. Depth-to-disparity conversion  
4. Pixel shifting to generate the second view  
5. Hole filling and rendering

---

# Project Background

The system demonstrates how RGB-D cameras can generate stereoscopic XR content in real time without requiring a dual-camera stereoscopic setup.

The technology has been demonstrated in multiple scenarios including:

- VR headset visualization
- 3D monitor displays
- live XR content streaming
- robotics tele-operation visualization

---

# Troubleshooting

### Camera not detected

Check Windows **Device Manager** to verify the Orbbec camera driver is installed correctly.

---

### Headset not recognized

Open the Oculus Desktop App and confirm the headset appears under **Devices**.

---

### No video in VR

Make sure:

- Quest Link is enabled
- The RGBD2VR application is running
- The camera stream is active
