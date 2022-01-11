---
layout: post
title: "[Project] Crack Detection 전처리(1)"
date: 2022-01-11 16:35 +0800
categories : DL
tags: [project, 설치에러]
toc:  true
math : True
---

## 데이터 수집 경로

---

- **도로장애물/표면 인지 영상(수도권)**
    
<a style="display: block;text-align: center;" href="https://aihub.or.kr/aidata/34111">
    <!-- <center> -->
        <img src="https://github.com/chaaaning/chaaaning.github.io/blob/master/images/%EB%8F%84%EB%A1%9C%EC%86%90%EC%83%81_%EC%A0%84%EC%B2%98%EB%A6%AC1/AI_HUB.png?raw=true">
    <!-- </center> -->
</a>
<center><span style="color:silver">사진을 누르면 이동합니다</span></center>
## 데이터 구성

---

### **데이터 구조**

---

- **Training, Validation 데이터**
    
    ![data_structure_crop.png](https://github.com/chaaaning/chaaaning.github.io/blob/master/images/%EB%8F%84%EB%A1%9C%EC%86%90%EC%83%81_%EC%A0%84%EC%B2%98%EB%A6%AC1/data_structure_crop.png?raw=true)
<br><br>

### 데이터 설명

---

1. **Annotations**
    - Crack 및 Bounding Box의 정보를 담고 있는 데이터 (Crack Detection이므로 Crack 정보만 활용)
    - 기본 형식은 `.json` 형식
        ```text
        Annotations.json - - - - - - - - dictionary load
          |— info
          |   |— description
          |   |— url
          |   |— version
          |   |— year
          |   |— contributor
          |   |— data created
          |  
          |— images - - - - - - - - - - - - - dictionary
          |   |— file name
          |   |— height
          |   |— width
          |   |— id
          |   
          |— annotations - - - - - - - - - - list (type(list[0])==dict())
          |   |— segmentation
          |   |— polyline - - - - - - - - - array
          |   |— image_id
          |   |— bbox
          |   |— id
          |   |— category_id
          |   |— area
          |   |— is_crowd
          |   |— id
          |     
          |— categories - - - - - - - - - list (type(list[0])==dict())
              |— id
              |— name
        ```
    - Crack Detection에서는 `annotations` 의 `polyline` 을 활용
    - `polyline` 은 [[$x_1,\,y_1,\,x_2,\,y_2,\cdots,x_n,\,y_n$]]의 형태이므로 reshape하여 사용해야 함
2. **Images**
    - 도로의 Crack을 보여주는 이미지 데이터
    - U-Net의 Input으로 사용
    - 기본 형식은 `.png` 형식으로 `cv2` 또는 `pillow` 패키지를 통해서 운용
3. **CRACK**
    - 도로의 균열에 관한 데이터
    - 해당 프로젝트의 메인 데이터 셋으로 사용됨
4. **TOA**
    - Object Detection에 사용되는 데이터
    - `yolo v5` 기반으로 annotation으로 `bbox (bounding box)` 로 객체가 정의됨
        
        ![Untitled](https://github.com/chaaaning/chaaaning.github.io/blob/master/images/%EB%8F%84%EB%A1%9C%EC%86%90%EC%83%81_%EC%A0%84%EC%B2%98%EB%A6%AC1/Untitled.png?raw=true)
        
5. **Type Of Images**
    - Frontback : 앞, 뒤의 블랙 박스 화면과 유사한 앵글의 이미지 데이터
    - Highway : 고속도로 이미지 데이터
    - Industrialroads : 공사현장, 산업단지, 공단 주변 도로 이미지 데이터
    - Kidzone : 어린이 보호구역 이미지 데이터
    - Mainroad : 국도 및 지방도 이미지 데이터
<br><br>

## 활용 데이터

---

해당 프로젝트는 Crack Detection으로 도로의 손상을 확인할 수 있는 모델 생성을 목적으로 한다. 그러므로 Crack에 관련한 데이터를 모델에 활용한다. Crack에 활용되는 데이터는 다음과 같다.

**이미지** : `./Training/Images/CRACK/`  **어노테이션** : `./Training/Annotations/CRACK/`

TOA 데이터는 필요에 따라 활용하고, Local 저장소의 문제로 한 번에 다운 받을 수 없다. 따라서 Crack의 어노테이션을 활용해서 전처리를 수행한다.
<br><br>

## 전처리 상세

---

Carvana(kaggle) 모델을 사용하기 위해서는 이미지 데이터와 마스킹 데이터가 필요하므로 Annotation을 마스킹 데이터와 같이 픽셀화 해야 함. 픽셀화를 위한 아이디어는 2가지

> **방법 1. plt.plot()**
> 
1. `.json` 파일 읽어오기
2. 720 X 1280 크기의 검정 색공간 생성 (numpy)
3. `.json` 파일에서 `key : annotations` 안에 `polylines`를 차례대로 읽어옴
4. `pyplot`의 `figure` 공간에 `polyline` 그리기
5. `plt.savefig()`를 통해 `.png` format으로 저장

**[방법 1의 특징]**

- 검정 색공간과 Crack Annotation을 객체화하여 저장하는 방식
- plt 객체를 저장하기 때문에 크기 조정이 까다로움
- 실수 공간의 좌표를 사용하기 때문에 픽셀에 0과1이 공존할 수 있음
- 용량이 크고 오래 걸림

> **방법 2. cv2.polylines()**
> 
1. `.json` 파일 읽어오기
2. 720 X 1280 크기의 검정 색공간 생성 (numpy)
3. `.json` 파일에서 `key : annotations` 안에 `polylines`를 차례대로 읽어옴
4. `cv2.polylines()`를 활용해 crack 그리기
5. `cv2.imwrite()`를 통해 `.png` format으로 저장

**[방법 2의 특징]**

- cv2로 polyline을 즉시 그릴 수 있음
- 픽셀 공간이 numpy를 활용하므로 annotation의 polyline 좌표값의 미세한 손실 발생 (반올림 사용)
- 정수 공간의 좌표를 사용하므로 픽셀에 0 또는 1만 존재
- 빠른 처리 보장
<span style="color:magenta">
→ *현재는 Carvana(kaggle) 모델을 기반으로 전처리하고 있지만, 모델과 input format에 따라 달라질 수 있음*
</span>