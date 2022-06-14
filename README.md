# [Lv2 P-Stage] Data Annotation / #눈#사람
> 📑 Wrapup Report 보러가기 [>> PDF]()

## Members
| 김하준 | 송민수 | 심준교 | 유승리 | 이창진 | 전영우 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| ![눈사람_김하준](https://user-images.githubusercontent.com/43572543/164686306-5f2618e9-90b0-4446-a193-1c8e7f1d77ad.png) | ![눈사람_송민수](https://user-images.githubusercontent.com/43572543/164686145-4030fd4f-bdd1-4dfa-9495-16d7c7689731.png) | ![눈사람_심준교](https://user-images.githubusercontent.com/43572543/164686612-d221b3c9-8895-4ac4-af4e-385412afe541.png) | ![눈사람_유승리](https://user-images.githubusercontent.com/43572543/164686476-0b3374d4-1f00-419c-ae5a-ecd37227c1ef.png) | ![눈사람_이창진](https://user-images.githubusercontent.com/43572543/164686491-c7acc30f-7175-4ce5-b2ea-46059857d955.png) | ![눈사람_전영우](https://user-images.githubusercontent.com/43572543/164686498-d251b498-b3fa-4c3c-b5f9-7cd2b62ed58b.png) |
|[GitHub](https://github.com/HajunKim)|[GitHub](https://github.com/sooya233)|[GitHub](https://github.com/Shimjoonkyo)|[GitHub](https://github.com/seungriyou)|[GitHub](https://github.com/noisrucer)|[GitHub](https://github.com/wowo0709)|

<br>

## Competition : 글자 검출 대회
<img width="1095" alt="스크린샷 2022-04-22 오후 3 06 23" src="https://user-images.githubusercontent.com/43572543/164680573-0b6cf69a-e073-4650-b999-472f5ffe7ea6.png">

### Introduction

스마트폰으로 카드를 결제하거나, 카메라로 카드를 인식할 경우 자동으로 카드 번호가 입력되는 경우가 있습니다. 또 주차장에 들어가면 차량 번호가 자동으로 인식되는 경우도 흔히 있습니다. 이처럼 OCR (Optimal Character Recognition) 기술은 사람이 직접 쓰거나 이미지 속에 있는 문자를 얻은 다음 이를 컴퓨터가 인식할 수 있도록 하는 기술로, 컴퓨터 비전 분야에서 현재 널리 쓰이는 대표적인 기술 중 하나입니다.

OCR task는 글자 검출 (text detection), 글자 인식 (text recognition), 정렬기 (Serializer) 등의 모듈로 이루어져 있습니다. 본 대회는 아래와 같은 특징과 제약 사항이 있습니다.

- 본 대회에서는 '글자 검출' task 만을 해결하게 됩니다.
- **Input** : 글자가 포함된 전체 이미지
- **Output** : bbox 좌표가 포함된 UFO Format
- **Model** : EAST (An Efficient and Accurate Scene Text Detector) 

### Metric
> DetEval 방식

1) 모든 정답/예측박스들에 대해서 Area Recall, Area Precision을 미리 계산
2) 모든 정답 박스와 예측 박스를 순회하면서, 매칭이 되었는지 판단하여 박스 레벨로 정답 여부를 측정
3) 모든 이미지에 대하여 Recall, Precision을 구한 이후, 최종 F1-Score은 모든 이미지 레벨에서 측정 값의 평균으로 측정

<br>

## Dataset

- ICDAR17_Korean
- ICDAR17_MLT
- ICDAR19_MLT
- AIStages

<br>

## Project Structure

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

## How to Run

### Installation
    
```bash
pip install -r requirements.txt

apt-get update
apt-get install ffmpeg libsm6 libxext6  -y

pip install mlflow
```
    
### Train
    
```bash
python train.py --experiment_name --validation 1
```

- `--experiment_name [실험이름]` : mlflow experiment를 생성합니다.
- `--validation 1` : validation을 수행합니다.

<br>

## LB Score Chart

<img width="371" alt="변동추이" src="https://user-images.githubusercontent.com/43572543/164683780-9b6c29c3-51fd-420d-b7fa-2542dc155028.png">

### Final LB Score
- **[Public]** F1-Score : 0.6583, Recall : 0.5724, Precision : 0.7745
- **[Private]** F1-Score : 0.6377, Recall : 0.5630, Precision : 0.7352
