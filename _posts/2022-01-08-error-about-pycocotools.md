---
layout: post
title: "[설치에러] pycocotools 패키지는 왜 설치가 안될까?"
date: 2022-01-08 01:24 +0800
categories : ERROR
tags: [error, pycocotools, 설치에러]
toc:  true
---

## pycocotools는?
---
pycocotools는 obeject detection, face detection 등 어떤 객체를 검출하는 vision 분야 딥러닝에서 접할 수 있습니다. pycocotools는 COCO(*Comon Object COntext*)의 `.json` 파일을 손쉽게 다룰 수 있는 패키지입니다. 실제로 COCO의 데이터 셋 뿐만 아니라, 검출 계열에서 annotation으로 많이 제공됩니다.
<a style="display: block;text-align: center;" href="https://cocodataset.org/#download">
    <center>
        <img src="https://github.com/chaaaning/chaaaning.github.io/blob/master/images/%EC%84%A4%EC%B9%98%EC%97%90%EB%9F%AC_pycocotools/coco.png?raw=true">
    </center>
</a>
<center>
    <span style="color:silver">
    사진을 누르면 COCO Dataset으로 이동합니다
    </span>
</center>
<br>

위의 사진에 COCO의 데이터셋 링크를 첨부해두었습니다. Object Detection을 공부할 때 좋은 데이터들이 많습니다. 이러한 COCO의 json에 대해서는 [아직 없음]("https://chaaaning.github.io")에 더 자세히 기술해 보겠습니다.
<br><br>

## pycocotools의 설치
---
여러 번의 오류를 겪으면서 설치할 수 있는 네 가지 방법 정도를 알게 되었습니다. 이는 다음과 같습니다.

1. `pip install pycocotools`
2. `conda install -c conda-forge pycocotools`
3. `pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"`
4. **`PyPl에서 직접 설치`**



결론부터 말하자면 저는 4번의 방법으로 성공했습니다. 다 실패하고 쓰는 글이기 때문에 실패 과정의 이미지는 없습니다.😞 우선 저는 생소한 패키지를 설치할 때, 무작정 `pip install packages`를 하기 보다는 구글에 `pycocotools install`과 같은 식으로 검색을 합니다. 그럼 보통 conda나 PyPl에서 제공하는 커맨드를 입력합니다.
<br><br>

### `pip install pycocotools`
---
해당 커맨드는 제 기억으로 `Visual Studio 14+` 버전 설치에 대한 에러가 출력되었습니다. 따라서 `Visual Studio 14+` 빌드 도구까지 설치해 보았지만 해결되지 않았습니다.
<br><br>

### `conda install -c conda-forge pycocotools`
---
웬만해서는 실패율이 거의 없는 커맨드가 conda 커맨드입니다. 구글에 검색했을 때 보시면, 

<center>
    <img src="https://github.com/chaaaning/chaaaning.github.io/blob/master/images/%EC%84%A4%EC%B9%98%EC%97%90%EB%9F%AC_pycocotools/conda_install.png?raw=true">
</center>
<center><span style="color:silver">conda 홈페이지에서 제공하는 커맨드</span></center>


이렇게 총 4개의 커맨드를 제공하지만, 모두 다 **찾을 수 없는 패키지**라는 경고를 반환합니다. conda 내에 구축이 되어있지 않다면 다운 받을 수 없다는 생각이 들어 포기했습니다.
<br><br>

### `pip install "git+https://github.com/philferriere...`
---
이 방식은 pycocotools를 git에서 직접 가져오는 방식입니다. 이것 역시 `Visual Studio 14+` 버전 설치로 인해 실패했습니다.

### `PyPl에서 직접 설치`
---
패키지를 직접 설치하는 것은 본인의 로컬 PC 환경만 잘 맞추어서 설치하면 가장 안전한 방법입니다. 이전에 Geopandas 패키지를 설치할 때에도 이것 저것 많이 알아보다가 결국 오류가 있는 사전 패키지들을 설치하면서 해결했습니다.

<a style="display: block;text-align: center;" href="https://pypi.org/project/pycocotools/">
    <center>
        <img src="https://github.com/chaaaning/chaaaning.github.io/blob/master/images/%EC%84%A4%EC%B9%98%EC%97%90%EB%9F%AC_pycocotools/pypl_pycocotools.png?raw=true">
    </center>
</a>
<center>
    <span style="color:silver">
    사진을 누르면 PyPl pycocotools로 이동합니다
    </span>
</center>
<br>

사진에서 보시면 PyPl은 `pip install pycocotools` 커맨드를 제공하지만 잘 실행되지 않았습니다. 따라서 왼쪽 사이드 바의 <span style="color:navy">Download files</span>를 클릭하여 패키지 파일을 다운 받습니다. 압축을 풀고 본인이 편한 디렉토리에 두시면 됩니다. 저의 경우 `C:\Users\user\`에 두었습니다. 그리고 cmd창을 실행합니다.(윈도우 키 누르고 cmd 검색)

> **경로 이동**

```cmd
절대경로 : cd C:/Users/user/pycocotools-2.0.3
상대경로 : cd ./pycocotools-2.0.3
```
위와 같이 압축해제된 `pycocotools-2.0.3`이 있는 디렉토리로 이동합니다. 상대 경로 사용 시 `.`을 찍으면 현재 경로를 의미합니다.즉, `C:/Users/user/ = ./`

> **설치 실행**

```cmd
python setup.py install
```
위와 같이 실행을 하고 jupyter나 vscode에 가셔서 `import pycocotools`를 실행하여 확인해 보면 잘 수행되는 것을 알 수 있습니다.
<br><br>

## 마치며
---
최근 딥러닝을 하면서 이것 저것 막히는게 굉장히 많아졌습니다. 블로그를 시작할 때만 하더라도 이전에 막혔던 vscode 환경설정, geoPandas 등등 하려고 했는데 당분간 딥러닝에서 문제가 많이 발생할 것 같습니다. 뭐가 됐든 꾸준히 기록해 보겠습니다. 감사합니다.

{% if page.comments %}
<div id="post-disqus" class="container">
{% include disqus.html %}
</div>
{% endif %}