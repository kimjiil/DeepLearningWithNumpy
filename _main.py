

'''
Model을 module에서 선언하고 외부에서 __call__ 할경우 자동으로 forward함수를 타게됨
이때, 각각의 레이어에 대해서도 똑같이 __call__ 할경우 -> forward()

model에서 .backward()를 호출할경우 gradient 계산

optimizer에서 .step() 호출하경우 gradient update
optimizer 선언부에서 model.paprameters()


sequential class는 module class에서 받아와서 module의 add_module을 실행해서 Layer를 추가함.

'''