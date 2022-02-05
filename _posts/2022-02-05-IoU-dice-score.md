---
layout: post
title: "[image] Intersection over Union(IoU) 와 Dice Score"
date: 2022-02-05 18:50 +0900
categories : DL
tags: [image]
toc:  true
math: true
---

딥러닝을 간략히 이야기 해보자면 신경망을 앞 뒤로 오가면서 최적의 가중치를 신경망마다 부여하여 학습을 시키는 것입니다. 간략하지 않은 것 같기는 하지만 단순하게 일단 정답을 외쳐보고 정답이랑 얼마나 벗어나 있는지 평가하여 다시 정답을 외치는 일을 반복하면서 최대한 정답에 가깝게 외치도록 만드는 것이지요. 글로는, 말로는 이런 식으로 표현할 수 있지만 수식과 코드로 이를 풀어 나가는데 있어서 "정답이랑 얼마나 벗어나 있는가"에 대해서 Loss Function이라는 손실 함수를 사용합니다. 손실 함수는 학습하고자 하는 데이터와 프로젝트에 따라 다른 함수를 사용합니다. 예를 들어 시계열 데이터라고 하면 RMSE, MSE와 같은 잔차에 대한 손실 함수, 분류 타입에 대해서는 Cross Entropy, Binary Cross Entropy 등의 함수를 사용하게 됩니다. 일단 해당 포스팅은 손실 함수의 이론적인 설명보다 이미지에서 사용되는 IoU와 Dice Score라는 이미지 데이터에서 Sementic Segmentation, Object Detection에 사용되는 손실함수를 다루고자 합니다.

## IoU(Intersection over Union)
---
IoU, Intersection over Union은 그대로 해석해보면 합집합 위의 교집합이라는 뜻입니다. 글자 그대로 풀이한다면 다음과 같은 수식으로 표현할 수 있습니다.
<center>
$$
y = {Intersection \over Union}
$$
</center>
그렇다면 두 개의 개별 집합이 존재한다는 것인데 이는 정답과 예측값입니다.

<br><br>

{% if page.comments %}
<div id="post-disqus" class="container">
{% include disqus.html %}
</div>
{% endif %}