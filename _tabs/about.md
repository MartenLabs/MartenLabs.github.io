---
layout: page
title: "About Me"
icon: fas fa-user
order: 1
---

## 📌 **Overview**

AI Systems Engineer specializing in end-to-end AI system development—from lightweight model design to scalable infrastructure deployment. Experienced in deploying ultra-lightweight thermal object detection on MCUs and designing secure server architectures for air-gapped environments.

**Highlights:**
- Published as first author in **IEEE Access (SCIE)** and presented two papers at **IEEE ICCE 2025**
- Filed **2 domestic patents** and **1 international PCT patent**
- Experience across the full stack: AI models, embedded systems, backend services, and cloud infrastructure

> [ORCID](https://orcid.org/0009-0004-0998-6643): 0009-0004-0998-6643

<br/>

---

## 📌 **Professional Experience**

### **A.I.Matics** — AI Systems Developer *(Sep 2025 ~ Present)*

**Tech Stack:** C/C++, Python, TensorRT, TFLite, Docker, Embedded Linux, OpenCV

I optimize AI models for edge devices and design system architectures for real-world deployment.

#### Key Projects

**🔹 Mobile-Conn Gateway Application** — *Deployed on Roadscope R11*
- Built a C++ gateway exposing internal system state via HTTP(S) REST API
- Implemented ZeroMQ IPC, cpp-httplib REST server, and HTTPS certificate security
- Delivered APIs for camera/ADAS/sensor configuration, firmware management, and device provisioning
- **Successfully deployed in production**

**🔹 Zero-Touch Server Architecture for Restricted Networks** — *[Patent Pending: 1st Inventor]*
- Designed an automated deployment system: **USB insert → full service ready in ~17 minutes**
- Key technologies:
  - **cloud-init + autoinstall** for zero-touch OS and GPU setup
  - **LXC + Docker** nested virtualization for complete service isolation
  - **L2 network integration** with direct DHCP/ARP and mDNS support
  - **WireGuard/Firewall** auto-configuration and TPM-based integrity checks
- Built offline package pipelines for air-gapped environments
- **Pending delivery to Dreamtech**

---

### **한국인물검증(Korea Personnel Verification Company)** — Backend Developer *(Dec 2021 ~ Dec 2022)*

**Tech Stack:** Spring Boot, Flask, AWS, Swagger, Jenkins

**Project:** KKAYO — Personal Information Verification System
- Built diploma verification engine and Spring Boot API server
- Managed AWS cloud infrastructure
- **Results:** 3x faster response times, 30% reduction in server costs

<br/>

---

## 📌 **Research**

### **FLARE: Real-Time Thermal Object Detection** *(2024)*

Developed an ultra-lightweight detection model for low-resolution thermal cameras.

- **118x smaller** memory footprint and **12x faster** than YOLOv8n
- Runs in real-time on **STM32 MCU**
- Custom augmentation techniques achieving **95% mAP**

**Tech:** Python, TFLite, OpenCV, YOLO, MobileNetV2-SSD, STM32, X-CUBE-AI

---

### **ObjectBlend: Data Augmentation for Imbalanced Datasets** *(2024)*

Created a novel augmentation technique for industrial defect detection.

- Improved YOLOv3-tiny mAP50-95: **0.20 → 0.91**
- Outperforms CutMix and similar methods
- Significantly boosts minority class (defect) detection

**Tech:** Python, OpenCV, PyTorch, YOLO, NVIDIA AGX Orin, TensorRT

<br/>

---

## 📌 **Publications**

**Journal**
- **IEEE Access (SCIE)** — *Real-Time Object Detection Using Low-Resolution Thermal Camera for Smart Ventilation Systems*  
  [DOI: 10.1109/ACCESS.2025.3566635](https://ieeexplore.ieee.org/document/10982063)

- **IEEE Access (SCIE)** — *ObjectBlend: Data Augmentation Technique for Vision Inspection Systems* *(Under Review)*

**Conference**
- **IEEE ICCE 2025** — *Real-Time Object Detection Using Low-Resolution Thermal Camera*  
  [DOI: 10.1109/icce63647.2025.10930159](https://ieeexplore.ieee.org/document/10930159)

- **IEEE ICCE 2025** — *ObjectBlend: Data Augmentation Technique for Vision Inspection Systems*  
  [DOI: 10.1109/icce63647.2025.10929866](https://ieeexplore.ieee.org/document/10929866)

<br/>

---

## 📌 **Patents**

**Korea**
- *Low-Resolution Thermal Object Detection and Real-Time Inference* — **10-2024-0127351**
- *Thermal Imaging Data Augmentation Algorithm* — **10-2024-0127352**

**International (PCT)**
- *Autonomous Ventilation System Using Predictive Model* — **PCT/KR2025/014712**

<br/>

---

## 📌 **Skills**

| Category | Technologies |
|----------|-------------|
| **AI/ML** | TensorFlow, PyTorch, YOLO, MobileNetV2-SSD, TFLite, TensorRT |
| **Embedded** | STM32 (CubeMX, X-CUBE-AI), Raspberry Pi, NVIDIA Jetson |
| **Backend & Cloud** | Spring Boot, Flask, FastAPI, AWS, Docker |

---

🔗 **GitHub:** [github.com/MartenLabs](https://github.com/MartenLabs)  
🔗 **Blog:** [MartenLabs](https://martenlabs.github.io/about/)

