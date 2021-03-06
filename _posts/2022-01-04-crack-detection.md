---
layout: post
title: "[Project] Road Crack Detection 프로젝트를 시작하며"
date: 2022-01-04 11:54 +0800
categories : DL
tags: [project]
toc:  true
---

## 새로운 프로젝트
---
이전에 머신러닝과 통계에만 치중된 프로젝트를 진행했기에 새로운 프로젝트로 딥러닝 프레임워크를 사용할 수 있는 프로젝트 주제를 선정했습니다. 마지막으로 딥러닝을 활용했을 때는 Face Detection을 학부 과제로 수행했을 때인데, 당시에 사용하던 i5 8세대의 내장 그래픽만 있는 노트북으로 어떻게 했는지 모르겠네요. (진짜 고집 부리면서 로컬로 돌렸던 기억이..)

지금은 나름 RTX3060이 달린 노트북이 있기도 하고, 꽤나 코딩에 익숙해졌기 때문에 정말 재미있게 할 수 있을 것 같습니다. 우선은 다음의 단계로 프로젝트를 할 계획입니다.

{% highlight text %}

1. Train Data 수집
2. Test Data 수집
3. 데이터 전처리 모델
4. 손상 검출 모델
5. 검증 모델

{% endhighlight %}
<br><br>

## 앞으로..
---
뭐 아직은 이정도로 구상만 해놓은 단계이지만, 다행스럽게도 같이 진행할 팀원도 있습니다. 또 기대되는 점은 gpu 기반의 모델 개발환경을 세팅하고, 최근 맛들린 github를 통한 협업도 기대가 됩니다. 벌써부터 부딪힐 문제들이 무섭고도 걱정이 되지만,, 꾸준히 기록하며 프로젝트 성공적으로 마무리 해보겠습니다!

{% if page.comments %}
<div id="post-disqus" class="container">
{% include disqus.html %}
</div>
{% endif %}