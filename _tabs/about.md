---
layout: page
title: "About Me"
icon: fas fa-user
order: 1
---

## 📌 **개요**
경량 AI 모델과 백엔드 시스템 개발 역량을 보유하고 있으며, MCU 및 엣지 디바이스 환경에서도 실시간 추론이 가능한 저해상도 열화상 객체 탐지 모델을 구현했습니다. IEEE Access(SCIE) 1저자 저널 게재, IEEE ICCE 2025 국제 컨퍼런스 발표 2편, 국내 특허 2건 및 국제 특허 1건 출원, Spring Boot 기반 REST API 및 클라우드 인프라 운영 경험을 통해 AI 시스템 전체 파이프라인을 설계·구축할 수 있는 능력을 갖추었습니다.

> [ORCID](https://orcid.org/0009-0004-0998-6643) : 0009-0004-0998-6643

<br/>
<br/>

---

## 📌 **주요 경력**

### 1. **한국인물검증 (2021.12 ~ 2022.12 / 1년)**

#### **프로젝트: 개인정보 검증 시스템(KKAYO)**
- **기술 스택:** Spring Boot, Flask, AWS, Swagger, Jenkins
- **역할:** 백엔드 개발, AWS 인프라 구축 및 운영
- **주요 업무:**
  - 대학교 졸업증명서 원본 확인 엔진 개발
  - Spring Boot 기반 API 서버 개발 및 외부 서비스 연동
  - AWS 기반 클라우드 인프라 구축 및 운영
- **성과:**
  - 프로젝트 전체 아키텍처 설계 및 백엔드 개발 주도
  - 데이터 자동 수집 및 API 최적화를 통해 시스템 응답 속도 **3배** 향상
  - AWS 비용 최적화 전략을 적용하여 서버 운영 비용 30% 절감

---

### 2. **A.I.Matics (2025.09 ~ 현재 / AI Systems Developer)**

- **기술 스택:** C/C++, Python, TensorRT, TFLite, Docker, Embedded Linux, OpenCV
- **역할:** AI 모델 최적화 및 시스템 아키텍처 설계

- **주요 업무:**
  - MCU/엣지 디바이스 환경에서 **실시간 추론 가능한 AI 모델 포팅 및 경량화**
  - TensorRT, TFLite 기반 모델 최적화 및 성능 개선
  - Docker 기반 배포 환경 설계 및 성능 검증

- #### **주요 프로젝트:**
  ##### **Mobile-Conn 게이트웨이 애플리케이션 (Roadscope R11 장비 탑재)**
  - **개요:** Core App 내부 상태/설정을 외부(모바일/상위시스템)에 **HTTP(S) REST API**로 안전하게 제공하는 C++ 기반 게이트웨이
  - **구성:** ZeroMQ IPC 내부 통신 · cpp-httplib 기반 REST 서버 · Problem+JSON 오류 처리 · HTTPS 인증서 기반 보안
  - **성과:**  
    - 카메라/ADAS/센서/저장장치 설정 조회 및 변경 API 구현  
    - 초기 설정·프로비저닝 및 펌웨어 인증서 관리 기능 개발  
    - Peripheral 테스트 및 주행거리 관리 기능 적용  
    - Roadscope R11 **최신 장비 상용 탑재**
  
    
  ##### **제한된 네트워크 환경 자동 구성형 학습 서버 아키텍처 구축 (2025.09 ~ 진행 중)** **[특허 출원 진행 중]**

  * **개요:**
    폐쇄망·격리망 등 외부 인바운드 접속이 제한된 환경에서 **USB 1개 삽입만으로 OS 설치 → 보안 설정 → 네트워크 구성 → 서비스 기동까지 자동 수행**할 수 있는 Zero-Touch 기반 학습 서버 자동 배포·운영 시스템을 설계 및 구현.

  * **구성:**

    * cloud-init/autoinstall 기반 **OS 자동 설치 파이프라인**
    * Self-Healing 로직 일부 구현(오류 감지, Retry, TPM Hash 검증 등) — *개발 진행 중*
    * **LXC 기반 가상화 환경** + L2 Public Bridge 구성
    * **LXC**(시스템 컨테이너) 내 **Docker**(애플리케이션 컨테이너) ***중첩 구성(Nested Virtualization)** 을 통한 서비스 격리 및 보안성 강화
    * WireGuard + stunnel + Split DNS 구성
    * FastAPI 기반 학습 제어 API & SSE 실시간 로그 스트리밍
    * Prometheus + Grafana 모니터링 환경
    * NVIDIA 드라이버 자동 설치 및 GPU 환경 자동 구성

  * **성과:**
  
    * **Zero-Touch 배포:** USB 삽입만으로 OS 설치·초기 보안·네트워크·컨테이너 실행까지 자동화
    
    * **L2 기반 네트워크 통합 구성:**
      * LXC 컨테이너가 물리 LAN과 동일한 L2 도메인에서 동작
      * 상위 라우터의 DHCP/ARP 직접 처리
      * mDNS·Broadcast 기반 서비스 탐색 정상 동작
      * macvlan 방식의 제한(호스트–컨테이너 통신 불가) 해결
    
    * **Self-Healing 구성 일부 적용 (개발 진행 중):**
      * 오류 발생 시 자동 Retry
      * TPM Hash 검증 기반 무결성 점검
      * Rollback 구조 설계 완료
    
    * **보안 운영 자동화:**
      * WireGuard 정책 자동 등록
      * 내부 CA 기반 TLS/stunnel 구성
      * nftables/UFW 기반 방화벽 정책 자동 생성
    
    * **컨테이너 기반 서비스 격리:**
      * Backend/Frontend를 LXC 내부 Docker Compose로 분리
      * 호스트 환경 오염 방지 및 유지보수성 향상
  
    * **운영 안정성 및 편의성 향상:**
      * FastAPI 기반 학습 제어 REST API
      * 실시간 SSE 로그 스트리밍
      * 주요 취약점(XSS, Path Traversal, Command Injection) 제거
  
    * **초기화 속도 최적화:**
      * OS 설치 약 10분 + Docker Compose 빌드 5~6분
      * 전체 초기화 과정 **약 17분 자동 완료**

    * **특허 출원 예정:**
      * *「제한된 네트워크 환경에서의 컴퓨팅 노드 자동 배포 및 보안 운영 시스템」*
        (주발명자, 변리사 검토 완료)
    
    * **드림텍 납품 대기중**


<br/>
<br/>

---
## 📌 **연구 및 기술 성과**

### 1. **FLARE: 저해상도 열화상 실시간 객체 탐지 모델** *(2024.01 \~ 2024.12 / 1년)*

- **기술 스택:** Python, TensorFlow Lite, OpenCV, YOLO, MobileNetV2-SSD, CubeMX, X-CUBE-AI, Raspberry Pi, STM32
- **주요 성과:**

  - YOLOv8n 대비 **메모리 사용량 1/118, 추론 속도 12배 향상**
  - STM32 MCU 환경에서 **초경량 실시간 객체 탐지 모델 구현**
  - **저해상도 열화상 전용 증강 기법 개발**, mAP 95% 성능 달성


### 2. **ObjectBlend: 데이터 불균형 해결 증강 기법** *(2024.07 \~ 2024.11 / 5개월)*

- **기술 스택:** Python, OpenCV, PyTorch, YOLO, NVIDIA AGX Orin, TensorRT
- **주요 성과:**

  - YOLOv3-tiny의 mAP50-95 **0.1988 → 0.9147**로 성능 대폭 향상
  - 기존 증강 기법(CutMix 등) 대비 **우수한 정밀도 확보**
  - **소수 클래스(결함)** 탐지 성능 대폭 개선

<br/>
<br/>

---

## 📌 **논문 및 학술 성과**

- **IEEE Access (SCIE 저널) · 1저자 게재**

  - *Real-Time Object Detection Using Low-Resolution Thermal Camera for Smart Ventilation Systems* <br/>
    [DOI: 10.1109/ACCESS.2025.3566635](https://ieeexplore.ieee.org/document/10982063)

- **IEEE Access (SCIE 저널) · 1저자, 심사 중**

  - *ObjectBlend: Data Augmentation Technique for Vision Inspection Systems*

<br/>

- **IEEE ICCE 2025 (국제 컨퍼런스) · 1저자 발표**

  - *Real-Time Object Detection Using Low-Resolution Thermal Camera for Smart Ventilation Systems* <br/>
    [DOI: 10.1109/icce63647.2025.10930159](https://ieeexplore.ieee.org/document/10930159)

  - *ObjectBlend: Data Augmentation Technique for Vision Inspection Systems* <br/>
    [DOI: 10.1109/icce63647.2025.10929866](https://ieeexplore.ieee.org/document/10929866)

<br/>
<br/>

---

## 📌 **특허 출원**

### 국내
- *저해상도 열화상 기반 객체 탐지 및 실시간 추론 기술*
  특허 출원번호: **10-2024-0127351**

- *열화상 이미지 전용 데이터 증강 알고리즘*
  특허 출원번호: **10-2024-0127352**

### 국제(PCT)
- *예측 모델을 활용한 자율 조절 환기 시스템을 구동하기 위한 전자 장치 및 그 구동 방법*  
  국제 특허 출원번호: **PCT/KR2025/014712**  

<br/>
<br/>

## 📌 **기술 역량**

- **AI/딥러닝:** TensorFlow/PyTorch, YOLO 시리즈, MobileNetV2-SSD, TFLite, TensorRT, 모델 경량화, 데이터 증강
- **Embedded AI:** Raspberry Pi, STM32(CubeMX, X-CUBE-AI), NVIDIA Jetson
- **백엔드 및 클라우드:**  Spring Boot, Flask, FastAPI, AWS, Docker, Swagger

---
🔗 **GitHub:** [github.com/MartenLabs](https://github.com/MartenLabs)  
🔗 **블로그:** [MartenLabs](https://martenlabs.github.io/about/)

