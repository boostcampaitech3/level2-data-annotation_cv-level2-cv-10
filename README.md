# [Lv2 P] Data Annotation :: #눈#사람

## 대회 설명
<img width="1095" alt="스크린샷 2022-04-22 오후 3 06 23" src="https://user-images.githubusercontent.com/43572543/164680573-0b6cf69a-e073-4650-b999-472f5ffe7ea6.png">

스마트폰으로 카드를 결제하거나, 카메라로 카드를 인식할 경우 자동으로 카드 번호가 입력되는 경우가 있습니다. 또 주차장에 들어가면 차량 번호가 자동으로 인식되는 경우도 흔히 있습니다. 이처럼 OCR (Optimal Character Recognition) 기술은 사람이 직접 쓰거나 이미지 속에 있는 문자를 얻은 다음 이를 컴퓨터가 인식할 수 있도록 하는 기술로, 컴퓨터 비전 분야에서 현재 널리 쓰이는 대표적인 기술 중 하나입니다.

OCR task는 글자 검출 (text detection), 글자 인식 (text recognition), 정렬기 (Serializer) 등의 모듈로 이루어져 있습니다. 본 대회는 아래와 같은 특징과 제약 사항이 있습니다.

- 본 대회에서는 '글자 검출' task 만을 해결하게 됩니다.
- **Input** : 글자가 포함된 전체 이미지
- **Output** : bbox 좌표가 포함된 UFO Format

## 데이터셋 설명

- ICDAR17_Korean
- ICDAR17_MLT
- AIStages
- ICDAR19_MLT

## 프로젝트 구조

> 본 repository의 `./data` 디렉토리를 제외한 파일은 아래 프로젝트 구조 중 `code` 디렉토리 내에 위치하는 파일이다.


```
.
|-- code
|   |-- OCR_EDA.ipynb
|   |-- convert_mlt.py
|   |-- custom_aug.py
|   |-- dataset.py
|   |-- detect.py
|   |-- deteval.py
|   |-- download_ICDAR.sh
|   |-- east_dataset.py
|   |-- inference.py
|   |-- loss.py
|   |-- model.py
|   |-- requirements.txt
|   |-- seed.py
|   |-- train.py
|   |-- urls.txt
|   `-- utils
|       |-- 01-aistages-ann-download.ipynb
|       |-- 02-cross-validation.ipynb
|       |-- 03-new-data.ipynb
|       |-- 04-combine-json.ipynb
|       `-- 05-cross-validation-full.ipynb
`-- input
    `-- data
        |-- AIStages_ANN
        |   |-- dataset
        |   `-- ufo
        |-- ICDAR17_Korean
        |   |-- images
        |   `-- ufo
        |-- ICDAR17_MLT
        |   |-- images
        |   |-- raw
        |   `-- ufo
        |-- ICDAR19_MLT
        |   |-- images
        |   |-- raw
        |   `-- ufo
        `-- Modified
            |-- images
            `-- ufo
```

## 실행 방법

- **라이브러리 설치**
    
    ```bash
    pip install -r requirements.txt
    
    apt-get update
    apt-get install ffmpeg libsm6 libxext6  -y
    
    pip install mlflow
    ```
    
- **학습 시작**
    
    ```bash
    python train.py --experiment_name --validation 1
    ```
    
    - `--experiment_name [실험이름]` - mlflow experiment를 생성한다.
    - `--validation 1` - validation을 수행한다.