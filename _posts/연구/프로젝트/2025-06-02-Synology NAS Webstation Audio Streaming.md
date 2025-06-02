---
title: Synology NAS Webstation Audio Streaming
date: 2025-06-02T9:00:00+09:00
categories: [연구, 프로젝트]
tags:
  [
    CORS, MIME, 오디오 스트리밍, web station
  ]
pin: true
math: true
mermaid: true
share: true 
comments: true
---

<br/>

### 상황

Synology NAS Webstation을 통해 오디오 스트리밍을 시도하면 Safari에서는 재생이 되지만 Chrome에서는 CORS 오류로 인해 오디오 스트리밍이 정상적으로 동작하지 않는 문제가 발생한다.

<br/>

### 문제 원인

1. NAS에서 **CORS 헤더를 반환하지 않거나**, **OPTIONS 프리플라이트 요청을 처리하지 않음**
    
2. 브라우저에서 오디오 파일의 **MIME 타입을 제대로 인식하지 못함**
    - 오디오 MIME 타입이 정확히 지정되어 있지 않으면, 브라우저에서 재생이 불가능할 수 있음

<p align="center"><img src="https://tera.dscloud.me:8080/Images/Project/CORSFIX/1.png" width="70%" height="70%"></p>

<br/>

### 해결 방법 (CORS 설정 추가)

#### 1. nginx 설정 파일(server.webstation-vhost.conf) 수정

``` bash
SynologyNAS:~$ cd /usr/local/etc/nginx/sites-enabled

SynologyNAS:/usr/local/etc/nginx/sites-enabled$ ls
server.pkg-static.Calendar-3924706297.conf  server.webstation.conf        synowstransfer-nginx.conf
server.ReverseProxy.conf                    server.webstation-vhost.conf
```

**server.webstation-vhost.conf** 파일을 찾았다면, `vim` 또는 `nano`로 수정한다.


수정 내용 예시:

``` vim
server {

    listen      xxx ssl default_server;       # 서비스 중인 포트 맞는지 확인
    listen      [::]:xxx ssl default_server;  # 서비스 중인 포트 맞는지 확인

    server_name _;

    include ...

    include ...

    add_header  
    ssl_prefer_server_ciphers   on;

    include ...

    include ...

    root    "...";
    index    ... , ..., ... ;

	##################### 여기부터 #####################
    location / {
    add_header Access-Control-Allow-Origin *;
    add_header Access-Control-Allow-Methods 'GET, POST, OPTIONS';
    add_header Access-Control-Allow-Headers 'Content-Type, Authorization';

        if ($request_method = 'OPTIONS') {
            add_header Access-Control-Max-Age 1728000;
            add_header Content-Type 'text/plain charset=UTF-8';
            add_header Content-Length 0;
            return 204;
        }
    }

    types {
        audio/wav wav;
    }
	##################### 여기 내용 추가 #####################
	
    include /usr/local/etc/nginx/conf.d/0572d98f-73c9-4658-bbb5-ae4eeb4d4d45/user.conf*;
}
```

<br/>


#### 2. nginx 설정 적용

설정을 저장했으면 아래 명령어로 nginx 구성을 확인

``` bash
sudo nginx -t   // nginx 재실행 
```


에러 없이 통과된다면, nginx를 재시작해 적용

``` bash
sudo nginx -s reload
```

<br/>

#### 3. 브라우저 캐시 초기화 및 확인
브라우저에서 새로고침 ( **Ctrl + Shift + R** ) 후
**F12 → 네트워크 탭**에서 요청/응답 헤더를 확인해보면 CORS 헤더가 정상적으로 반환되는지, MIME 타입 설정이 적용되었는지 확인할 수 있다.

<p align="center"><img src="https://tera.dscloud.me:8080/Images/Project/CORSFIX/3.png" width="70%" height="70%"></p>




