---
layout: post
title: my test post
author: rnch
tags: [test, mypost, math]
categories : Markdown
date: 2022-01-01 15:45 +0800
toc:  true
---
# 이건 첫 번째 포스팅입니다.
---
## 첫 번째 제목
콜아웃은 이렇게 쓰면 되나보네요
{: .message }
1. 포스팅이 되는가
2. 레이아웃 형성이 어떻게 되는가
3. 다음이 어떻게 동작하는가  

```markdown
layout: post
title: my test post
author: rnch
tags: [test, mypost, math]
date: 2022-01-01 15:45 +0800
```
## 두번째 제목
마크다운을 쓰는건 참으로 재미있는 일이다.
> 마크다운을 쓰세요 여러분  

이렇게 인용구도 써보고 이번 test post에서 알아봐야할 사항은 TOC가 동작 잘 하는지야  
  

# 이건 두 번째 대 제목입니다.
---
여기에는 하이라이팅을 해보겠습니다  
## 하이라이팅
{% highlight text %}
여러분 새해 복 많이 받으세요
{% endhighlight %}

이거는 페이지를 들어가야 확인할 수 있어요  
  

# 테이블
---
## 테이블을 만들겠습니다.
|1번|2번|3번|  
|---|---|---|  
|a|b|c|  
|d|e|f|  
  
마크다운 문법은 표에서는 적용이 되지 않습니다
{: .message}
## 테이블 마크다운 코드
```markdown
|1번|2번|3번|  
|---|---|---|  
|a|b|c|  
|d|e|f|  
```  
## 표를 만들려면 다음과 같이 만들어야 합니다  
<table>
  <thead>
    <tr>
      <th>첫번째</th>
      <th>두번째</th>
      <th>세번째</th>
    </tr>
  </thead>
  <tfoot>
    <tr>
      <td>내용1</td>
      <td>내용2</td>
      <td>내용3</td>
    </tr>
  </tfoot>
  <tbody>
    <tr>
      <td>그리고1</td>
      <td>그리고2</td>
      <td>그리고3</td>
    </tr>
    <tr>
      <td>행1</td>
      <td>행2</td>
      <td>행3</td>
    </tr>
    <tr>
      <td>열1</td>
      <td>열2</td>
      <td>열3</td>
    </tr>
  <tfoot>
    <tr>
      <td>열1</td>
      <td>열2</td>
      <td>열3</td>
    </tr>
  </tfoot>
  </tbody>
</table>