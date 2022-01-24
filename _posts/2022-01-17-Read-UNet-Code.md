---
layout: post
title: "[Project] Pytorch를 활용한 UNet Code 읽기(1)"
date: 2022-01-17 17:16 +0900
categories : DL
tags: [project]
toc:  true
---

우선 자세한 내용은 많이 생략되었지만, Crack Detection을 위해서는 <span style="color:magenta">UNet</span>이라는 모델을 활용해야 합니다. 이와 관련해서 찾아본 바로 Kaggle의 Carvana Dataset을 활용하여 자동차 모델의 마스킹을 생성하는 예제가 존재합니다. UNet 자체를 이해하는 것도 좋지만, 아직은 `torch`가 어색하기 때문에 기존에 있는 Code([Carvana UNet 관련 github](https://github.com/milesial/Pytorch-UNet))의 `data_loading.py` 코드를 읽어 보면서 제 것으로 만들어 보겠습니다.

 그 중 첫 번째로 읽어봐야 할 부분은 Data를 불러오는 부분입니다. `torch`는 dataset을 class로 구현하여 customize 해서 사용하는 경우가 많습니다. 따라서 오늘은 UNet의 해당 부분을 읽어보겠습니다.(생소한 모듈 및 함수 부분을 확인하면서 읽는 방식으로 진행해 보겠습니다.)
<br><br>

## 클래스 구성

---

`torch`에서는 데이터셋을 클래스의 형태로 커스터 마이즈할 수 있습니다. 해당 파이썬 코드에서 구현된 클래스의 구성은 다음과 같습니다.
```text
BasicDataset
 ├── Instance Variable
 │    ├── self.images_dir
 │    ├── self.masks_dir
 │    ├── self.scale
 │    ├── self.masks_suffix
 │    └── self.ids
 │
 ├── Instance Method
 │    ├── self.__init__()
 │    ├── self.__len__()
 │    └── self.__getitem__()
 │
 └── Class Method
      ├── preprocess()
      └── load()

CarvanaDataset ⇒ BasicDataset을 상속하는 형태      
```
인스턴스 변수는 초기자`__init__()`에 선언 되어 있고, [`torch`의 튜토리얼](https://tutorials.pytorch.kr/beginner/data_loading_tutorial.html?highlight=%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%85%8B)에 따르면 커스텀 데이터 셋을 구성할 때 `torch.utils.data.Dataset`을 상속하고 `__init__`, `__len__`, `__getitem__`을 오버라이드해야 한다고 합니다. 이에 따라 세 가지 메서드가 정의 되어 있고 특이하게 Class Method가 정의되어 있습니다.
<br><br>


## Module Import

---

```python
import logging
from os import listdir
from os.path import splitext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
```

 > **`import logging`**

로깅은 어떤 소프트웨어가 실행될 때 발생하는 이벤트를 추적하는 수단입니다. 해당 파이썬 코드에서는 파라미터, 장치 등의 사전 정의 값을 추적하고 확인하는 역할로 활용됩니다. 자세한 내용은 [logging 참조 문서](https://docs.python.org/ko/3/howto/logging.html)를 확인하시면 세부 함수 및 활용 예시를 확인할 수 있습니다.

 >**`from os.path import splitext`**

 `.splitext()`는 확장자를 구할 때 사용됩니다. input으로는 경로 형식의 문자열이 주어지고, output으로는 튜플을 반환합니다.
 ```python
os.path.splitext("this/is/file/path/file_name.csv")

''' output '''
('this/is/file/path/file_name', '.csv')
 ```
위와 같이 (`경로+파일명`,`확장자`) 형태로 문자열을 분리하고, 인덱싱을 통해 확장자에 접근할 수 있게 됩니다. 해당 파이썬 코드에서는 `Images`데이터와 `Masks`데이터가 파일명은 같지만, 확장자가 다른 점을 처리하기 위해 사용됩니다.

> **`from pathlib import Path`**

 이전까지는 경로를 `str`로 받아서 처리를 했었는데, 해당 코드에서는 경로를 객체로 두고 처리하게 됩니다. 즉 어떤 하나의 class로 정의된 영역에 경로 객체를 두고 객체의 인스턴스 변수와 메서드를 활용하여 핸들링하게 됩니다. `Path`는 경로를 객체화 시켜주는 모듈입니다. 이는 활용하는 환경에 따라 유동적으로 적용할 수 있도록 객체로 바꾸어 사용되는 것으로 보입니다. 자세한 내용은 [pathlib.Path의 참조 문서](https://docs.python.org/ko/3/library/pathlib.html)를 확인해 주시길 바랍니다.
<br><br>

## Class 내부 읽기
---
### `__init__(self)`
---
```python
def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = ''):
    self.images_dir = Path(images_dir)      
    self.masks_dir = Path(masks_dir)

    ''' pin 1 '''
    assert 0 < scale <= 1, 'Scale must be between 0 and 1' 

    self.scale = scale
    self.mask_suffix = mask_suffix

    ''' pin 2 '''
    self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]

    if not self.ids:
        raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

    logging.info(f'Creating dataset with {len(self.ids)} examples')
```

인스턴스 변수를 초기화하는 부분입니다. 생성되는 변수들에 대한 세부적인 내용은 다음과 같습니다.

- `images_dir` : image file이 있는 경로입니다.
- `masks_dir` : mask file이 있는 경로입니다. (label에 해당)
- `scale` : 이미지 사이즈를 조절할 스케일을 지정합니다.
- `mask_suffix` : mask file의 파일명이 image file의 뒤에 어떠한 suffix가 붙어 있을 경우 명시합니다.
- `ids` : 파일을 탐색할 파일 이름들을 담은 리스트입니다.

이렇게 정리할 수 있고, 세부적인 구문은 다음과 같은 구문들이 존재합니다.

> **`assert`**

파이썬을 입문하던 시기에 지나가는 문법으로서 본 적이 있었던 `assert`문입니다. 이는 <span style="color:red">가정설정문</span>이라고 불리우고 역할은 조건 검사 후 AssertionError를 반환하는 것입니다. 기본적인 형태는 `assert (조건), "message"`의 형태로 조건을 만족하지 못하면 AssertionError를 출력합니다. *`pin 1`*에 해당하는 구문을 보면 `scale`이 0과1 사이에 있지 않으면 "스케일은 0과 1 사이에 있어야 합니다."라는 안내문을 반환하게 됩니다. 이러한 `assert`구문은 실수가 유발 될 수 있는 부분에 넣어 사용할 때 유용할 것으로 보입니다. 자세한 내용은 [asset 문에 대한 문서](https://wikidocs.net/21050)를 참조하시면 좋을 것 같습니다.

> **`str.startswith(self)`**

지정한 문자열로 시작하면 True, 그렇지 않다면 False를 반환합니다. *`pin 2`*의 구문에서는 `.`으로 시작하고 있으면 True를 반환하므로, `.`으로 시작하지 않으면 `splitext(file)[0]`로 동작하여 파일명을 받아오게 됩니다.
```example
>> example
file = "file_name.csv"
splitext(file) -> tuple("file_name",".csv")
splitext(file)[0] -> "file_name"
```
<br>

### `__len__(self)`
---
```python
def __len__(self):
    return len(self.ids)
```
`ids` 변수의 길이를 반환합니다. 즉, 데이터 셋으로 사용되는 데이터의 갯수를 반환하는 것으로 이해할 수 있습니다.
<br>

### `preprocess()`
---
```python
@classmethod
def preprocess(cls, pil_img, scale, is_mask):
    w, h = pil_img.size
    newW, newH = int(scale * w), int(scale * h)
    assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'

    """ pin 1 """
    pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
    img_ndarray = np.asarray(pil_img)

    """ pin 2 """
    if img_ndarray.ndim == 2 and not is_mask:
        img_ndarray = img_ndarray[np.newaxis, ...]
    elif not is_mask:
        img_ndarray = img_ndarray.transpose((2, 0, 1))

    if not is_mask:
        img_ndarray = img_ndarray / 255

    return img_ndarray
```

`__getitem__()`에서 사용될 class method 입니다. class method에 대한 내용은 [해당 페이지](https://wikidocs.net/16074)를 확인해 보시길 바랍니다.

`preprocess()`메서드는 의미 그대로 이미지 데이터를 전처리 합니다. 상단부에서 입력된 `scale`에 맞게 끔 이미지 크기를 재조정하고, 옳지 않을 경우 Assertion Error를 반환합니다.

중단부인 *`pin 1`* 은 입력받은 데이터를 이미지 리샘플링을 해줍니다.

마지막으로 하단부인 *`pin 2`* 는 mask 데이터와 image 데이터를 각각 다르게 처리하는데, mask 데이터는 그대로 사용되고, image 데이터의 경우 2차원인 경우에는 `np.newaxis`를 통해 차원을 넓혀줍니다. 이는 `torch.tensor`에서 이미지 데이터가 (channel, width, height)를 의미하므로 채널 차원을 추가한다고 생각할 수 있습니다. 2차원이 아닌 경우에는 numpy의 형태를 바꿔 주는데, numpy는 (width, height, channel)로 이미지를 받아 오기 때문에 tensor에 맞게끔 channel을 가장 앞으로 가져오는 작업을 수행합니다. 그리고 255의 색상값을 1로 정규화하기 위해 255로 나누어 주는 작업을 실행합니다.

> **`pillow.Image.resize()`**

이미지 데이터는 대표적으로 `pillow` 라이브러리와 `cv2` 라이브러리를 활용합니다. 그 중 `Image.resize()`는 사이즈를 조절하는 것과 더불어, 이미지 리샘플링을 진행합니다.[패키지의 공식문서](https://pillow.readthedocs.io/en/stable/reference/Image.html)에 정의된 바에 따르면 `Image.resize()`는 다음과 같이 정의 되어 있습니다.
```python
Image.resize(size, resample=None, box=None, reducing_gap=None)
```
이 중, 해당 코드에서 활용 된 파라미터는 `size`와 `resample`로서 *`pin 1`*라인에서 `size`는 튜플의 형태로 들어가기 때문에 스케일 변환이 완료된 width와 height가 들어갑니다. 그리고 `resample`에 해당하는 옵션으로 두 가지가 주어지는데, mask 데이터에 대해서는 `Image.NEAREST`를 적용하고 image 데이터에 대해서는 `Image.BICUBIC` 옵션을 적용합니다.
- `Image.NEAREST` : 입력 이미지에서 가장 가까운 픽셀을 선택하고, 다른 모든 입력 픽셀은 무시합니다.
- `Image.BICUBIC` : 크기 조정 시 출력에 기여하는 픽셀 값을 보간하는 하나의 방식으로 삼차 보간법을 사용하여 픽셀을 리샘플링합니다.

> **`[np.newaxis, ...]`**

당황했던 구문입니다. 우선 대괄호가 있기 때문에 인덱싱의 기법이라고 추측할 수 있었습니다. 더 정확히 알아 본 결과, `np.newaxis`는 차원을 늘릴 때 사용하는 인스턴스입니다. 직관적으로는 대괄호를 추가하는 명령어 입니다. 여기서 `...`이 의미하는 바는 평소 넘파이나 판다스를 인덱싱할 때 자주 사용되는 `:`과 같은 역할을 합니다.

> **`np.transpoase()`**

numpy의 전치행렬을 취해주는 함수입니다. 대표적으로 `np.T`가 있지만 이는 단순히 행과 열을 바꿔주는 반면, `np.transpose()`는 파라미터에 따라 원하는 형태로 `numpy`를 변화시킬 수 있습니다. *`pin 2`*의 구문을 보면 파라미터 값이 (2, 0, 1)입니다. 이는 본래 존재하는 2행을 가장 앞으로 보내겠다는 것을 의미합니다. 즉 (r1, r2, r3)에 대해 (r3, r1, r2)의 형태로 바꾸는 것입니다. 이는 `numpy`와 `tensor`가 이미지 데이터에서 의미하는 바가 다르기 때문입니다.
- `numpy` : channel last (width, height, channel)
- `torch.tensor` : channel first (channel, width, height)

### **`load()`**
---
```python
@classmethod
def load(cls, filename):
    ext = splitext(filename)[1]
    if ext in ['.npz', '.npy']:
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)
```
실제로 이미지를 읽어 오는 메서드입니다. `PIL.Image`의 형태로 읽어오고 확장자 맞게 끔 읽어 올 수 있도록 역할을 수행합니다.

### **`__getitem__(self)`**
---
```python
def __getitem__(self, idx):
    name = self.ids[idx]
    mask_file = list(self.masks_dir.glob(name + self.mask_suffix + '.*'))
    img_file = list(self.images_dir.glob(name + '.*'))

    assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
    assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
    mask = self.load(mask_file[0])
    img = self.load(img_file[0])

    assert img.size == mask.size, \
        'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

    img = self.preprocess(img, self.scale, is_mask=False)
    mask = self.preprocess(mask, self.scale, is_mask=True)

    return {
        'image': torch.as_tensor(img.copy()).float().contiguous(),
        'mask': torch.as_tensor(mask.copy()).long().contiguous()
    }
```
해당 메서드는 불러온 이미지를 `Dataloader`에 올라갈 수 있게 끔 만들어 줍니다. `assert`문을 통해 데이터 변화 과정의 오류를 점검할 수 있습니다. 큰 흐름은
1. image, mask data의 파일명 불러오기
2. 파일명을 통해 데이터 불러오기
3. 불러온 데이터 전처리 하기

와 같은 과정을 띄고 있습니다.
<br><br>

## 마치며
---
데이터 불러오는 `.py` 파일을 읽어 보았습니다. 나름 파이썬에 익숙해졌다고 생각했는데, 기초 문법부터 `torch`, `numpy`까지 부족한 점을 많이 느낄 수 있었습니다. 그래도 코드 라인을 한 줄씩 읽으면서 `assert`문은 실제로 이 다음 변형 과정에서도 디버깅을 하는 데에도 유용하게 사용했습니다.

{% if page.comments %}
<div id="post-disqus" class="container">
{% include disqus.html %}
</div>
{% endif %}
