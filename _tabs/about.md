---
layout: page
title: "About Me"
icon: fas fa-user
order: 1
---

## 📌 **Overview**
AI Systems Engineer with expertise in lightweight AI model development and end-to-end system infrastructure design. I implemented an ultra-lightweight, low-resolution thermal object detection model capable of real-time inference on MCU and edge devices, and published as the first author in IEEE Access (SCIE) along with two additional first-author presentations at IEEE ICCE 2025. I also designed and built an industrial AI training server architecture for restricted and air-gapped environments, incorporating Zero-Touch provisioning, virtualized service isolation, and secure network integration. With two domestic patents and one international patent filed, combined with backend and cloud experience using Spring Boot, REST APIs, and cloud infrastructure, I am equipped to architect and deploy complete AI system pipelines—from models and services to networking, security, and automated deployment.

> [ORCID](https://orcid.org/0009-0004-0998-6643) : 0009-0004-0998-6643

<br/>
<br/>

---

## 📌 **Professional Experience**

### 1. **한국인물검증 (Dec 2021 ~ Dec 2022 / 1 Year)**

#### **Project: Personal Information Verification System (KKAYO)**
- **Tech Stack:** Spring Boot, Flask, AWS, Swagger, Jenkins
- **Role:** Backend Development, AWS Infrastructure Setup and Operations
- **Key Responsibilities:**
  - Developed university diploma original verification engine
  - Built Spring Boot-based API server and integrated external services
  - Established and operated AWS-based cloud infrastructure
- **Achievements:**
  - Led overall project architecture design and backend development
  - Improved system response speed by **3x** through automated data collection and API optimization
  - Reduced server operating costs by 30% by applying AWS cost optimization strategies

---

### 2. **A.I.Matics (Sep 2025 ~ Present / AI Systems Developer)**

- **Tech Stack:** C/C++, Python, TensorRT, TFLite, Docker, Embedded Linux, OpenCV
- **Role:** AI Model Optimization and System Architecture Design

- **Key Responsibilities:**
  - **Porting and optimizing AI models for real-time inference** on MCU/edge device environments
  - Model optimization and performance improvement based on TensorRT and TFLite
  - Designing Docker-based deployment environments and performance validation

- #### **Key Projects:**
  ##### **Mobile-Conn Gateway Application (Deployed on Roadscope R11 Equipment)**
  - **Overview:** C++ based gateway that securely exposes Core App internal state/settings to external systems (mobile/upper systems) via **HTTP(S) REST API**
  - **Components:** ZeroMQ IPC internal communication · cpp-httplib based REST server · Problem+JSON error handling · HTTPS certificate-based security
  - **Achievements:**  
    - Implemented APIs for camera/ADAS/sensor/storage device configuration queries and modifications  
    - Developed initial setup/provisioning and firmware certificate management features  
    - Applied peripheral testing and mileage management functionality  
    - **Commercially deployed on latest Roadscope R11 equipment**
  
    
  ##### **Auto-Configured Learning Server Architecture for Restricted Network Environments (Sep 2025 ~ Ongoing)** **[Patent Application in Progress - 1st Inventor]**

  * **Overview:**
    A Zero-Touch based automated server deployment system that **completes OS installation to service startup in approximately 17 minutes with just USB insertion** in restricted network environments.

  * **Key Achievements and Technologies:**
    * **Zero-Touch Deployment Automation:** 100% automation of OS installation, network/security configuration, and GPU environment setup using cloud-init and autoinstall.
    * **LXC+Docker Nested Virtualization:** Separated system (LXC) and application (Docker) layers to prevent host contamination and achieve complete service isolation.
    * **L2 Network Integration:** Configured same L2 domain as physical LAN for direct upstream router DHCP/ARP handling and mDNS service discovery support (overcoming macvlan limitations).
    * Built internal repository mirroring and offline package deployment pipeline for air-gapped environments
    * **Enhanced Security and Stability:** Automatic application of WireGuard/Firewall policies, TPM-based integrity verification, and self-healing logic for automatic recovery on failure.
    * **Patent and Commercialization:**
      * Patent pending: *"Automated Deployment and Secure Operation System for Computing Nodes in Restricted Network Environments"* (1st Inventor, patent attorney review completed)
      * **Pending delivery to Dreamtech**


<br/>
<br/>

---
## 📌 **Research and Technical Achievements**

### 1. **FLARE: Real-Time Object Detection Model for Low-Resolution Thermal Imaging** *(Jan 2024 \~ Dec 2024 / 1 Year)*

- **Tech Stack:** Python, TensorFlow Lite, OpenCV, YOLO, MobileNetV2-SSD, CubeMX, X-CUBE-AI, Raspberry Pi, STM32
- **Key Achievements:**

  - **1/118 memory usage and 12x faster inference speed** compared to YOLOv8n
  - Implemented **ultra-lightweight real-time object detection model** on STM32 MCU environment
  - Developed **augmentation techniques specialized for low-resolution thermal imaging**, achieving 95% mAP performance


### 2. **ObjectBlend: Data Augmentation Technique for Addressing Data Imbalance** *(Jul 2024 \~ Nov 2024 / 5 Months)*

- **Tech Stack:** Python, OpenCV, PyTorch, YOLO, NVIDIA AGX Orin, TensorRT
- **Key Achievements:**

  - Significantly improved YOLOv3-tiny mAP50-95 from **0.1988 → 0.9147**
  - Achieved **superior precision** compared to existing augmentation techniques (CutMix, etc.)
  - Dramatically improved detection performance for **minority classes (defects)**

<br/>
<br/>

---

## 📌 **Publications and Academic Achievements**

- **IEEE Access (SCIE Journal) · First Author Publication**

  - *Real-Time Object Detection Using Low-Resolution Thermal Camera for Smart Ventilation Systems* <br/>
    [DOI: 10.1109/ACCESS.2025.3566635](https://ieeexplore.ieee.org/document/10982063)

- **IEEE Access (SCIE Journal) · First Author, Under Review**

  - *ObjectBlend: Data Augmentation Technique for Vision Inspection Systems*

<br/>

- **IEEE ICCE 2025 (International Conference) · First Author Presentation**

  - *Real-Time Object Detection Using Low-Resolution Thermal Camera for Smart Ventilation Systems* <br/>
    [DOI: 10.1109/icce63647.2025.10930159](https://ieeexplore.ieee.org/document/10930159)

  - *ObjectBlend: Data Augmentation Technique for Vision Inspection Systems* <br/>
    [DOI: 10.1109/icce63647.2025.10929866](https://ieeexplore.ieee.org/document/10929866)

<br/>
<br/>

---

## 📌 **Patent Applications**

### Domestic (Korea)
- *Low-Resolution Thermal Imaging-Based Object Detection and Real-Time Inference Technology*
  Patent Application No.: **10-2024-0127351**

- *Data Augmentation Algorithm Specialized for Thermal Imaging*
  Patent Application No.: **10-2024-0127352**

### International (PCT)
- *Electronic Device and Driving Method for Operating Autonomous Adjustable Ventilation System Using Predictive Model*  
  International Patent Application No.: **PCT/KR2025/014712**  

<br/>
<br/>

## 📌 **Technical Skills**

- **AI/Deep Learning:** TensorFlow/PyTorch, YOLO Series, MobileNetV2-SSD, TFLite, TensorRT, Model Optimization, Data Augmentation
- **Embedded AI:** Raspberry Pi, STM32(CubeMX, X-CUBE-AI), NVIDIA Jetson
- **Backend & Cloud:** Spring Boot, Flask, FastAPI, AWS, Docker, Swagger

---
🔗 **GitHub:** [github.com/MartenLabs](https://github.com/MartenLabs)  
🔗 **Blog:** [MartenLabs](https://martenlabs.github.io/about/)

