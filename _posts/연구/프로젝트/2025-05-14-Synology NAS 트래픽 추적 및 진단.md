---
title: 서버 내부 트래픽 추적 및 진단
date: 2025-05-14T9:00:00+09:00
categories: []
tags:
  [
    서버, 네트워크, 트래픽, 추적, 진단, NAS, Synology NAS
  ]
pin: true
math: true
mermaid: true
share: true 
comments: true
---

# 서버 내부 트래픽 추적 및 진단

> ⚠️ 리눅스 표준 패키지 매니저(apt, yum 등)를 사용할 수 없는 Synology NAS 환경을 대상으로 함.

<br/>

---
### 목표

- 지속적인 NAS 업로드 트래픽 (예: 1MB/s)의 원인을 분석한다.

- 실시간 연결 상태와 프로세스를 추적하는 방법을 정리한다.

- 문제가 발생했을 때 즉시 사용할 수 있는 명령어와 분석 기준을 제공한다.

<br/>
<br/>

---
## 기본 네트워크 점검

### 1. 연결된 IP/포트 및 프로세스 확인


`netstat` 은 현재 시스템이 열고 있는 모든 네트워크 연결 정보를 확인할 수 있다. Synology NAS에서 패키지 설치 없이도 기본 제공.


``` bash
netstat -ntp
```

#### 주요 옵션

| 옵션   | 설명                                         |
| ---- | ------------------------------------------ |
| `-n` | 호스트 이름을 **IP 주소로 출력** (DNS 조회 생략, 속도 향상)   |
| `-t` | **TCP 연결만** 필터링                            |
| `-p` | 연결을 사용하는 **프로세스 ID(PID)** 및 **프로그램 이름** 표시 |

<br/>

> ❗ 루트 권한이 없으면 일부 프로세스 정보는 표시되지 않을 수 있음.

#### 출력 예

``` bash
Active Internet connections (w/o servers)
Proto Recv-Q Send-Q Local Address           Foreign Address         State       PID/Program name
tcp        0      0 192.168.0.3:445         192.168.0.2:62408       ESTABLISHED 18309/smbd
tcp        0      0 192.168.0.3:36398       52.38.239.161:443       ESTABLISHED 32390/synorelayd
tcp        0      0 127.0.0.1:161           127.0.0.1:43528         ESTABLISHED 11324/snmpd
tcp        0      0 192.168.0.3:22          221.150.2.34:61030      ESTABLISHED 10866/sshd: [계정]
tcp        0    232 192.168.0.3:22          192.168.0.1:62374       ESTABLISHED 18156/sshd: [계정]
tcp        0      0 127.0.0.1:43528         127.0.0.1:161           ESTABLISHED 11400/synosnmpcd
```

#### 분석

| 항목                  | 설명                                                          |
| ------------------- | ----------------------------------------------------------- |
| **ESTABLISHED 상태**  |  현재 데이터 송수신이 실제로 이루어지고 있는 연결.<br/> → 이 상태의 외부 IP가 지속적인 업로드 트래픽의 대상일 수 있음                             |
| **포트 번호**           | 443(HTTPS), 80(HTTP), 22(SSH) 등 자주 쓰이는 포트 위주로 확인.      |
| **Foreign Address** | 외부 IP 주소가 **신뢰할 수 있는 서버인지** 확인 (ex. AWS, Cloudflare 등). |
| **PID/Program**     | 해당 연결을 사용 중인 프로세스를 식별해 어떤 앱에서 트래픽이 발생했는지 추적.<br/><br/>- 로 표시된다면<br/>→ 루트 권한이 없거나 해당 프로세스를 식별할 수 없는 상태<br/>→ sudo netstat -ntp 또는 ss -ntp 명령어로 루트 권한으로 재확인 권장 |

<br/>

#### 선택 사항

* 의심되는 IP는 `whois`, `ipinfo.io`, 또는 `abuseipdb.com` 등으로 조회

* 다수의 `22번 포트` 접속이 있다면, **SSH 브루트포스 공격** 가능성 고려

* 외부 IP와 장시간 연결되어 있고 PID 정보가 없으면 **숨겨진 프로세스**나 루트 권한 필요 여부 점검

특정 PID가 확인되면 다음 명령으로 프로세스 상세 정보 확인
``` bash
ps -p <PID> -o comm=
top -p <PID>
```


<br/>
<br/>
<br/>

---
### 2. 인터페이스 및 네트워크 환경 정보 확인

지속적인 업로드 트래픽 문제를 진단하려면, 우선 NAS가 어떤 네트워크 인터페이스를 통해 통신 중인지 정확히 파악해야 한다. 

이를 통해 이후 사용하는 트래픽 분석 도구(예: iftop, tcpdump, nethogs)에서 정확한 인터페이스를 지정할 수 있다.


``` bash
ip addr
ip link show
```

| 명령어            | 설명                                               |
| -------------- | ------------------------------------------------ |
| `ip addr`      | 장비에 할당된 **IP 주소**, 서브넷 정보, **인터페이스 활성 상태** 등을 확인 |
| `ip link show` | 모든 **네트워크 인터페이스의 물리적 상태** (UP/DOWN, MAC 주소 등) 확인 |


#### 목적

- NAS에서 실제로 사용 중인 네트워크 인터페이스(예: eth0, ovs_eth0, bond0)를 식별한다.

- 이후 트래픽 모니터링 시, 정확한 인터페이스 이름을 명시하여 분석 정확도를 높힌다.

- DOWN 상태의 비활성화 인터페이스, 가상 브리지(docker0 등)와 실제 트래픽 인터페이스를 명확히 구분한다.

<br/>

#### 출력 예

``` bash
$ ip addr

1: lo: <LOOPBACK,UP,LOWER_UP> mtu 65536 ...
    inet 127.0.0.1/8 scope host lo
2: eth0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 ...
    inet 192.168.0.3/24 brd 192.168.0.255 scope global eth0
3: docker0: <NO-CARRIER,BROADCAST,MULTICAST,DOWN> mtu 1500 ...
    inet 172.17.0.1/16 scope global docker0
4: ovs_eth0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc noqueue state UP group default qlen 1
    link/ether ...
    inet 192.168.0.3/24 brd 192.168.0.255 scope global ovs_eth0
       valid_lft forever preferred_lft forever
    inet6 ....
       valid_lft forever preferred_lft forever
```

``` bash
$ ip link show

5: ovs_eth0: <BROADCAST,MULTICAST,UP,LOWER_UP> mtu 1500 qdisc noqueue state UP mode DEFAULT group default qlen 1
    link/ether ...

```


#### 분석

| 항목              | 설명                                               |
| --------------- | ------------------------------------------------ |
| `state UP`      | 해당 인터페이스가 **현재 활성화**되어 있으며 트래픽 송수신 가능            |
| `<UP,LOWER_UP>` | **물리적으로 연결되어 있고 동작 중**인 상태                       |
| `inet` 필드       | IP가 할당되어 있으면 해당 인터페이스는 **네트워크 사용 중**             |
| 인터페이스 이름        | `iftop`, `tcpdump` 실행 시 `-i` 옵션으로 **정확히 지정해야 함** |


<br/>

#### 확인 사항

- Synology NAS는 일반 리눅스와 달리 다음과 같은 비표준 네이밍을 사용하는 경우가 많다:
  - ovs_eth0: Open vSwitch 기반의 가상 브리지
  - bond0: 두 개 이상의 물리 NIC를 묶은 링크 애그리게이션
  - veth*: Docker 컨테이너에서 생성된 가상 인터페이스

- 실시간 트래픽 분석 시 활성화된 물리 NIC(예: eth0, bond0 등) 만을 대상으로 해야 정확한 송수신량 측정이 가능하다.

<br/>
<br/>
<br/>


---
### 3. 실시간 트래픽 송수신 상태 확인

> ❗주의

| 항목          | 설명                                                            |
| ----------- | ------------------------------------------------------------- |
| Synology 환경 | 기본적으로 `apt`, `yum` 사용 불가                                      |
| 인터페이스 지정    | `iftop -i` 옵션에 앞서 확인한 실제 활성화된 인터페이스 입력 필요 (`eth0`, `bond0` 등) |

<br/>


지속적인 업로드 트래픽의 원인을 파악하기 위해, 어떤 외부 IP와 얼마나 많은 데이터를 주고받고 있는지를 실시간으로 확인하는 것이 중요하다. 

`iftop`은 트래픽 흐름을 직관적으로 보여주는 CLI 기반 도구이다.


``` bash
sudo synogear install iftop
iftop -i ovs_eth0
```

| 옵션            | 설명                                  |
| ------------- | ----------------------------------- |
| `-i [인터페이스명]` | 지정한 인터페이스의 트래픽만 모니터링 (예: `-i ovs_eth0`) |


#### 목적
- 실시간으로 어떤 IP가 데이터를 송수신 중인지 확인

- 업로드/다운로드 대역폭이 어느 정도인지 직접 확인

- 지속적인 외부 트래픽 유발 IP를 시각적으로 식별 가능

- 특정 포트 또는 서비스와의 고정적인 연결 유무 확인


<br/>

#### 출력 예

<p><img src="https://tera.dscloud.me:8080/Images/Network/서버_트래픽_추적_및_진단/iftop.png" width="100%" height="100%"></p>

#### 우측 대역폭 수치 해석

수치가 어떤 방향(업로드/다운로드)을 나타내는지는 **줄의 위치**에 따라 다름.

| 위치         | 의미            | 시간 기준    | 방향                             |
| ---------- | ------------- | -------- | ------------------------------ |
| **왼쪽 수치**  | 최근 **2초 평균**  | 가장 짧은 구간 | **해당 줄 기준 송신 (TX) 또는 수신 (RX)**<br/>**=>** : Outbound (송신 / 업로드)<br/>**<=** : Inbound (수신 / 다운로드)|
| **가운데 수치** | 최근 **10초 평균** | 중간 구간    | 송신/수신                          |
| **오른쪽 수치** | 최근 **40초 평균** | 가장 긴 구간  | 송신/수신                          |


#### 분석

| 항목                   | 설명                                                                     |
| -------------------- | ---------------------------------------------------------------------- |
| **TX(Transmit)** 대역폭 | 업로드 트래픽. **1MB/s 이상 지속적**인 항목은 주의 깊게 확인                                |
| **RX(Receive)** 대역폭  | 다운로드 트래픽. 패턴 파악용                                                       |
| **고정된 외부 IP**        | 특정 IP가 계속 대역폭을 차지한다면 **클라우드 백업, 악성 프로세스 등** 가능성 확인 필요                  |
| **연결 대상 IP**         | 외부 도메인/서비스 확인을 위해 IP → 도메인 역추적 (`whois`, `ipinfo.io`, `nslookup` 등) 필요 |

<br/>

#### 확인 사항

- iftop은 기본적으로 1, 5, 10초 단위 평균 트래픽을 함께 보여준다. (지속적인 트래픽 여부 파악에 유리)

- t, s, d, n 등 단축키로 정렬 조건, 표시방식 변경 가능

- 실제 파일 업로드가 없어도 동기화, 모니터링, 클라우드 서비스로 인해 업로드 트래픽이 발생할 수 있다.



<br/>
<br/>
<br/>

---
### 4. 특정 포트에 열려 있는 프로세스 확인

트래픽이 특정 포트를 통해 지속적으로 발생하고 있다면 어떤 프로세스가 해당 포트를 열고 있는지 확인하는 것이 중요하다. 특히 업로드 트래픽이 HTTPS(443번 포트) 기반이라면 lsof 명령어를 통해 관련 프로세스를 직접 추적할 수 있다.

``` bash
lsof -i :443
```

| 옵션     | 설명                                       |
| --------- | ---------------------------------------- |
| `lsof`    | 시스템에서 열린 파일(소켓 포함)을 나열하는 명령어             |
| `-i :443` | **포트 443**을 사용 중인 모든 네트워크 연결과 관련 프로세스 출력 |


#### 목적
- 443번 포트를 열고 있는 실행 파일 및 프로세스 식별

- netstat에서는 확인이 어려운 실행 파일 경로 및 커맨드 정보 확인

- 클라우드 싱크, 백업 서비스, 외부 API 통신 등의 실체 파악

- 의심스러운 지속 트래픽의 원인을 PID와 명령어 기반으로 추적

<br/>

#### 출력 예

``` bash
COMMAND     PID USER   FD   TYPE    DEVICE SIZE/OFF NODE NAME
nginx     12336 root   11u  IPv4     45097      0t0  TCP *:https (LISTEN)
nginx     12336 root   12u  IPv6     45098      0t0  TCP *:https (LISTEN)
nginx     31487 http   11u  IPv4     45097      0t0  TCP *:https (LISTEN)
nginx     31487 http   12u  IPv6     45098      0t0  TCP *:https (LISTEN)
nginx     31488 http   11u  IPv4     45097      0t0  TCP *:https (LISTEN)
nginx     31488 http   12u  IPv6     45098      0t0  TCP *:https (LISTEN)
synorelay 32390 root   14u  IPv4 151270921      0t0  TCP 192.168.0.3:36398->ec2-52-38-239-161.us-west-2.compute.amazonaws.com:https (ESTABLISHED)
```

#### 분석

| 항목        | 설명                                                              |
| --------- | --------------------------------------------------------------- |
| `COMMAND` | 해당 포트를 열고 있는 실행 프로그램 명 (예: `synology`, `cloud-sync`, `nginx` 등) |
| `PID`     | 프로세스 ID → `ps -p <PID>` 또는 `top -p <PID>`로 상세 조회 가능             |
| `USER`    | 해당 프로세스를 실행 중인 사용자 (예: `root`, `admin`, `nobody` 등)             |
| `FD`      | 파일 디스크립터(File Descriptor).<br/>네트워크 소켓일 경우 `u`(user), `t`(TCP) 등을 나타냄.|
| `DEVICE`  | 해당 파일 또는 소켓의 장치 식별자 (디바이스 번호) |
| `SIZE/OFF`| 열려 있는 파일이나 소켓의 사이즈 또는 오프셋.<br/>네트워크 연결의 경우 대부분 0t0으로 표시됨 (의미 없음) |
| `NAME`    | 통신 대상 외부 IP와 포트 정보 (예: `52.38.239.161:https`)                   |
| 상태       | `ESTABLISHED` 상태로 오래 유지 중이라면 **지속적인 데이터 송수신 중**일 가능성 높음         |

<br/>

#### 선택 사항 

- 포트별 점검 예시:

``` bash
lsof -i :22       # SSH 연결 확인
lsof -i :80       # HTTP 통신 확인
lsof -i :5000     # 사용자 지정 포트 확인
```


- PID 기반 상세 정보 확인:

``` bash
ps -p <PID> -o pid,ppid,cmd,etime
```

<br/>
<br/>
<br/>

---
## 추가 추적 도구 설치 (Entware + nethogs)

<br/>

### 1. Entware 설치

**Entware 란**

> [Entware](https://github.com/Entware/Entware/wiki/Install-on-Synology-NAS) Synology NAS에서 **고급 리눅스 패키지(`opkg`)를 설치 가능하게 해주는 환경**

<br/>

#### Entware 설치 (x86_64 기준)

> ❗root 계정으로 전환 후 실행

``` bash
sudo -i  # 필수
wget -O - https://bin.entware.net/x64-k3.2/installer/generic.sh | /bin/sh
```

<br/>

⚠️ 에러 발생 시

| 증상                                                    | 해결 방법                    |
| ----------------------------------------------------- | ------------------------ |
| `Permission denied`, `umount: /opt: target is busy` 등 | `/opt` 디렉토리 초기화 후 재설치 필요 |

``` bash
sudo -i
umount /opt 2>/dev/null
rm -rf /opt/*
```

→ 설치 명령 재실행

<br/>

#### 환경 변수 등록 (PATH 설정)

``` bash
echo 'export PATH=$PATH:/opt/bin:/opt/sbin' >> ~/.profile
source ~/.profile
```

<br/>

#### nethogs 설치

``` bash
opkg update
opkg install nethogs
```

<br/>
<br/>
<br/>

---
### 2. 트래픽 추적: nethogs

nethogs는 프로세스(PID) 단위로 실시간 네트워크 사용량을 추적하는 툴이다.

iftop이 IP 기반이라면, nethogs는 **어떤 프로세스가 얼마나 트래픽을 유발하는가**에 초점을 맞춘다.

``` bash
sudo nethogs ovs_eth0
```

| 옵션/인자      | 설명                                                   |
| ---------- | ---------------------------------------------------- |
| `[인터페이스명]` | 모니터링할 네트워크 인터페이스 지정 (예: `eth0`, `ovs_eth0`, `bond0`) |


#### 목적
- 트래픽 발생 주체를 실행 중인 프로세스 기준으로 확인

- IP 단위가 아닌 명확한 사용자/프로세스 기준 추적

- 클라우드 싱크, 도커, 자동 백업, 악성 프로세스 등 의심 대상 식별

<br/>

#### 출력 예

<p><img src="https://tera.dscloud.me:8080/Images/Network/서버_트래픽_추적_및_진단/nethogs.png" width="100%" height="100%"></p>


#### 분석

| 항목                  | 설명                                  |
| ------------------- | ----------------------------------- |
| `PID/USER/PROGRAM`  | 어떤 유저가 어떤 명령어를 통해 트래픽을 발생시키는지 확인 가능 |
| `Sent` / `Received` | 각 프로세스가 보낸/받은 트래픽을 실시간으로 표시         |
| 지속적으로 높은 전송량        | **자동 동기화, 외부 연동, 또는 이상 동작**의 가능성    |
| 특정 프로세스가 계속 상단 유지   | 해당 프로세스는 **주요 트래픽 유발자**일 가능성 높음     |


- nethogs는 관리자 권한 필수 (sudo)

- NAS에 Docker 환경이 있다면 docker-proxy, containerd, syno-cloud-syncd 등이 자주 등장할 수 있음

- 트래픽 많은 프로세스를 정리한 후 lsof -p <PID>, ps -p <PID> -o cmd,etime 등으로 정체 파악 추천

<br/>
<br/>

#### 기타 동작 중 프로세스 확인 및 해석


|프로세스|설명|끌 수 있는가|
|---|---|---|
|`nginx`|NAS 웹 UI 서버|⚠️ 끄면 DSM 웹 접속 불가|
|`sshd`|SSH 세션 (본인)|⚠️ 끄면 현재 세션 종료됨|
|`synorelayd`|QuickConnect 중계 서비스| QuickConnect 비활성화 시 자동 종료|
|`synoaic_a`|Active Insight (클라우드 리포트)| 설정에서 비활성화 가능|

<br/>
<br/>
<br/>

---
### 3. 패킷 추적: tcpdump

`tcpdump`는 실시간 네트워크 트래픽을 패킷 단위로 캡처하고 분석할 수 있는 가장 정밀한 도구 중 하나이다.

특정 포트, IP, 인터페이스를 기준으로 정밀 감시가 가능하며, 악성 통신, API 요청 추적 등에 활용된다.

<br/>

``` bash
sudo tcpdump -i ovs_eth0 port 443
```

| 옵션/필터           | 설명                                   |
| --------------- | ------------------------------------ |
| `-i ovs_eth0`   | 캡처할 인터페이스 지정 (`ip addr`에서 확인한 인터페이스) |
| `port 443`      | 특정 포트만 필터링 (HTTPS 통신)                |
| `host 1.2.3.4`  | 특정 IP와의 트래픽만 필터링                     |
| `-w trace.pcap` | 캡처 내용을 파일로 저장 (Wireshark로 분석 가능)     |
| `-c 100`        | 100개 패킷만 캡처 후 종료                     |


#### 목적
- 특정 포트/서비스에 대해 어떤 데이터가 오가는지 패킷 수준으로 확인

- 암호화되지 않은 프로토콜(HTTP, FTP 등) 은 데이터 내부까지 직접 확인 가능

- 캡처한 패킷을 Wireshark로 열어 상세 분석 가능

<br/>

#### 출력 예

``` bash
tcpdump: verbose output suppressed, use -v or -vv for full protocol decode
listening on ovs_eth0, link-type EN10MB (Ethernet), capture size 262144 bytes
00:19:41.988250 IP 192.168.0.3.36398 > 52.38.239.161.443: Flags [.], ack 1963333263, win 319, options [nop,nop,TS val 1097067264 ecr 3367404682], length 0
00:19:42.111090 IP 52.38.239.161.443 > 192.168.0.3.36398: Flags [.], ack 1, win 483, options [nop,nop,TS val 3367464843 ecr 1097041619], length 0
00:19:47.068293 IP 52.38.239.161.443 > 192.168.0.3.36398: Flags [.], ack 1, win 483, options [nop,nop,TS val 3367469797 ecr 1097041619], length 0
00:19:47.068360 IP 192.168.0.3.36398 > 52.38.239.161.443: Flags [.], ack 1, win 319, options [nop,nop,TS val 1097072344 ecr 3367464843], length 0
00:19:47.123456 IP 192.168.0.3.36398 > 52.38.239.161.443: Flags [P.], seq 123:456, ack 789, win 512, length 333

5 packets captured
10 packets received by filter
0 packets dropped by kernel
```

#### 분석

| 항목           | 설명                                            |
| ------------ | --------------------------------------------- |
| **IP/포트 정보** | 내 NAS의 어떤 포트가 외부의 어떤 포트와 통신 중인지 식별 가능         |
| **length**   | 전송된 데이터의 크기                                   |
| **Flags**    | \[S] 시작, \[F] 종료, \[P] 데이터 포함 여부 등 TCP 플래그 정보 |

<br/>

#### 선택 사항

- 로그 저장:

> 저장된 .pcap 파일은 Wireshark에서 열어 GUI 기반 분석 가능

``` bash
sudo tcpdump -i ovs_eth0 -w capture_443.pcap port 443
```

- 특정 IP 트래픽만 필터링:

``` bash
sudo tcpdump -i ovs_eth0 -n host 8.8.8.8 (ip 또는 lsof, iftop등 에서 확인한 주소)
```

- 실시간 전체 트래픽 확인:

``` bash
sudo tcpdump -i ovs_eth0 -n
```


<br/>
<br/>
<br/>
<br/>

---
## 분석 결과 

- 1MB/s 업로드는 **정상적인 내부 네트워크 트래픽** (ex. Time Machine 백업, Active Insight, QuickConnect 등)
    
- 외부로 나가는 의심 트래픽은 존재하지 않음



