# [Lv2 P] Data Annotation :: #눈#사람

## 팀원 구성
| [김하준_T3066](https://github.com/HajunKim) | [송민수_T3113](https://github.com/sooya233) | [심준교_T3124](https://github.com/Shimjoonkyo) | [유승리_T3129](https://github.com/seungriyou) | [이창진_T3169](https://github.com/JasonLee-cp-zz) | [전영우_T3192](https://github.com/wowo0709) |
|---|---|---|---|---|---|
| ![눈사람_김하준](https://user-images.githubusercontent.com/43572543/164686306-5f2618e9-90b0-4446-a193-1c8e7f1d77ad.png) | ![눈사람_송민수](https://user-images.githubusercontent.com/43572543/164686145-4030fd4f-bdd1-4dfa-9495-16d7c7689731.png) | ![눈사람_심준교](https://user-images.githubusercontent.com/43572543/164686612-d221b3c9-8895-4ac4-af4e-385412afe541.png) | ![눈사람_유승리](https://user-images.githubusercontent.com/43572543/164686476-0b3374d4-1f00-419c-ae5a-ecd37227c1ef.png) | ![눈사람_이창진](https://user-images.githubusercontent.com/43572543/164686491-c7acc30f-7175-4ce5-b2ea-46059857d955.png) | ![눈사람_전영우](https://user-images.githubusercontent.com/43572543/164686498-d251b498-b3fa-4c3c-b5f9-7cd2b62ed58b.png) |

<br>

## 대회 설명
<img width="1095" alt="스크린샷 2022-04-22 오후 3 06 23" src="https://user-images.githubusercontent.com/43572543/164680573-0b6cf69a-e073-4650-b999-472f5ffe7ea6.png">

스마트폰으로 카드를 결제하거나, 카메라로 카드를 인식할 경우 자동으로 카드 번호가 입력되는 경우가 있습니다. 또 주차장에 들어가면 차량 번호가 자동으로 인식되는 경우도 흔히 있습니다. 이처럼 OCR (Optimal Character Recognition) 기술은 사람이 직접 쓰거나 이미지 속에 있는 문자를 얻은 다음 이를 컴퓨터가 인식할 수 있도록 하는 기술로, 컴퓨터 비전 분야에서 현재 널리 쓰이는 대표적인 기술 중 하나입니다.

OCR task는 글자 검출 (text detection), 글자 인식 (text recognition), 정렬기 (Serializer) 등의 모듈로 이루어져 있습니다. 본 대회는 아래와 같은 특징과 제약 사항이 있습니다.

- 본 대회에서는 '글자 검출' task 만을 해결하게 됩니다.
- **Input** : 글자가 포함된 전체 이미지
- **Output** : bbox 좌표가 포함된 UFO Format
- **Model** : EAST (An Efficient and Accurate Scene Text Detector) 

<br>

## 사용한 데이터셋

- ICDAR17_Korean
- ICDAR17_MLT
- ICDAR19_MLT
- AIStages

<br>

## 프로젝트 구조

> 본 repository의 `./data` 디렉토리를 제외한 파일은 아래 프로젝트 구조 중 `code` 디렉토리 내에 위치하는 파일입니다.


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
|   |-- pths/
|   |-- requirements.txt
|   |-- seed.py
|   |-- train.py
|   |-- trained_models/
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

<br>

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
    
    - `--experiment_name [실험이름]` : mlflow experiment를 생성합니다.
    - `--validation 1` : validation을 수행합니다.

<br>

## 날짜별 LB Score 변동 추이

<img width="371" alt="변동추이" src="https://user-images.githubusercontent.com/43572543/164683780-9b6c29c3-51fd-420d-b7fa-2542dc155028.png">

#### 최종 점수
- **[Public]** f1 : 0.6583, recall : 0.5724, precision : 0.7745
- **[Private]** f1 : 0.6377, recall : 0.5630, precision : 0.7352
