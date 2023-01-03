# DeepLearningWithNumpy
Numpy로 구현한 DeepLearning

- 딥러닝 내부 알고리즘 및 학습 과정을 다시 공부하기 위해 작성중

- [블로그 내용 정리](https://kimjiil.github.io/ai/ml%20study/DeepLearningWithNumpy/)

---

## [DeepLearning Code with cupy, Numpy](./myLib)

- 기존 Numpy로 구성된 코드를 좀더 빠르게 GPU(cupy)에서 실행하기 위해서 작성함.
- Pytorch로 딥러닝 코드를 작성하는 것과 유사하게 하기 위해
Pytorch 함수의 이름과 기능을 비슷하게 작성중.


### Numpy, Numba, cupy 계산 시간 비교

```commandline
python MatMul_caclTime.py
```

| Library | device | rumtime        |
|---------|--------|----------------|
| Numpy | CPU | 0.007s ~ 0.01s |
| Numba | GPU | 0.04s ~ 0.06s | 
| cupy | GPU | 0.001s ~ 0.002s |


### 파일 구성
- myLib
  - Numpy, cupy로 작성한 deeplearning code
- main.py
  - myLib을 이용한 학습용 코드

### Train
```commandline
python main.py
```

- [[jupyter notebook]](https://github.com/kimjiil/DeepLearningWithNumpy/blob/main/notebooks/myDeepLearning%Code.ipynb)
  - MINST Dataset - Accuracy 99.82%
  - train data size - 60,000 / valid data size - 10,000 / batch_size - 144 / 1 epoch runtime - about 210s

---

## [DeepLearning Code with Numpy](./Lib_np)

- Numpy로만 작성한 초기 테스트 코드

### 파일 구성
- Lib_np 
  - Numpy로 작성한 deeplearning code
- main_np.py 
  - Lib_np을 이용한 학습용 코드

### Train
```commandline
python main_np.py
```

- [[jupyter notebook]](https://github.com/kimjiil/DeepLearningWithNumpy/blob/main/notebooks/DeeplearningWithNumpy_Training_Test.ipynb)
  - MINST Dataset - Accuracy 95.81%
  - train data size - 60,000 / valid data size - 10,000 / batch_size - 144 / 1 epoch runtime - 2hour

### Refrence
- https://github.com/SkalskiP/ILearnDeepLearning.py

--- 

