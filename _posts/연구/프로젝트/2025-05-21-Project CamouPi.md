---
title: Project CamouPi:A Self-Sovereign Stealth VPN & Threat Intelligence Node
date: 2025-05-21T9:00:00+09:00
categories: [연구, 프로젝트]
tags:
  [
    embedded-security, wireguard, luks-encryption, pi-hole,
    2fa-security, reinforcement-learning, anomaly-detection
  ]
pin: true
math: true
mermaid: true
share: true 
comments: true
---

<!-- ## Project CamouPi – Embedded. Encrypted. Intelligent.   -->
<!-- *A self-sovereign stealth VPN & threat intelligence node.* -->

*CamouPi – Embedded. Encrypted. Intelligent.*

---

**CamouPi**는 Raspberry Pi 4B(8GB) 기반의 개인 서버 프로젝트로 **WireGuard VPN**을 통해 외부 접근을 통제하고 내부 NAS 기능 및 보안 기능을 안전하게 운영할 수 있도록 설계되었다.

본 프로젝트는 단순한 홈서버를 넘어 **VPN을 통한 트래픽 암호화**, **LUKS 기반 파일 시스템 암호화**, **AI 기반 트래픽 분석 및 보안 정책 자동화**까지 확장되는 **지능형 보안 인프라 플랫폼**을 지향한다.

-  **WireGuard 기반 VPN 게이트웨이**
    
-  **암호화된 SFTP 전용 파티션 구축 (LUKS + 사용자 격리)**
    
-  **트래픽 패턴 기반 AI 보안 모델 개발 (LSTM + DQN)  [apply [OFA²](https://arxiv.org/abs/2303.13683)]**
    
-  **watchdog 자동 복구 시스템 설계**


**CamouPi**는 저사양 디바이스에서도 **보안성과 실용성을 모두 만족**시키기 위한 실험적 시도이며, 보안이 필수인 환경에서 **자율적인 보호와 제어가 가능한 서버 플랫폼**을 목표로 한다.

<br/>

``` txt
                             🌍 외부 네트워크 
                           (LTE, 도서관, 외부망)
                                   │
                             Phone / Laptop 
                           (WireGuard Client)
                                   │
                            🔐 WireGuard VPN 
                          (터널링, 10.8.0.0/24)
                                   │                                            
        ┌───────────────────────────────────────────────────────┐
        │                 Raspberry Pi 4B                         
        │------------------------------------------------------ │         
        │                                                       │                                   
            📦 OS: Raspberry Pi OS 8GB (2GHz OC)
            🔐 VPN: WireGuard Server 
            🌐 DNS: Pi-hole (DNS Filtering, Logging)
                - Upstream DNS: Cloudflare
                - Blocking List: StevenBlack + Customizable

            🔒 LUKS 암호화
                - 민감한 키, 인증 파일, 시드 securely 저장
        
            🤖 강화학습 보안 모듈
                - LSTM + DQN 기반 트래픽 패턴 임베딩 (OFA 도입)
                - scapy 패킷 단위 시계열 흐름 추출
                - 강화학습 정책	DQN/DDPG/TD3 등으로 행위 결정
                - pihole-FTL.db 활용 쿼리 패턴 학습
                - iptables, nftables, ufw 명령 자동 선택
        │                                                       │                                      
        └──────────────────────┬────────────────────────────────┘
                               │
                   🔁 로컬 네트워크 (192.168.0.0/24)
                               │
                ───────────────┴───────────────
                        📦 Synology NAS                     
                            - Blog image server 
                            - Drive Server 
                            - VSCode Server 
                            - Docker 
                            - Hyper Backup (→ 외장 SSD)  ─────  💾 외장 SSD (Samsung T5 EVO 8TB)         
                            - Time Machine 백업 공유                - NAS 백업 저장소
                            - ...                                 - Hyper Backup 암호화 백업 파일 저장
```             



