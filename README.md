# 의료 데이터에서 SSL을 위한 최적의 데이터 증강 기법 연구

## ⭐ 배경 및 주제
- 의료 데이터에서 SSL로 학습하기 위한 최적의 데이터 증강 기법을 연구하여 downstream task인 질환 분류에서 높은 성능을 달성하는 것을 목표로 한다. 
- Self-Supervised Learning(SSL)은 라벨 없이 데이터로부터 표현을 학습하는 기법으로, 특히 의료 데이터에서 중요성이 강조된다. 본 프로젝트를 통해 라벨링에 많은 시간과 비용이 소요되는 문제, 의료진 간 해석 차이로 인해 일관성이 부족한 문제, 희귀 질환이나 소아·노인 전용 데이터의 부족 문제, 환자 프라이버시와 데이터 보호 규정에 따른 데이터 활용 제한 등 다양한 제약이 존재하는 의료 데이터의 문제를 해결하고자 한다. 


## 💁 팀원
- **총 3인 : 권민지(팀장), 조현식, 황예은**


## 📅 진행 기간 
- **2024.11 ~ 2024.12 (2개월)**


## 🚀 파이프라인
![비전AI와비즈니스_최종발표](https://github.com/user-attachments/assets/b3bd18fd-3a3b-4783-b09d-4e7a2f04c4e5)
1. 두 개의 X-ray 데이터셋을 사용해 각 관절별로 정상과 비정상을 구분하는 이진 분류용 데이터셋을 구성한다.
2. 해당 데이터셋에서 데이터의 라벨을 삭제 후 다양한 데이터 증강 기법의 조합을 적용하여 SimCLR과 BYOL을 학습을 시킨다.
3. SimCLR 또는 BYOL Encoder를 Logistic Regression에 적용하여 분류 작업을 수행하고, 분류 성능 차이를 분석하여 최적의 데이터 증강 기법 조합이 무엇인지 찾아낸다.
4. SSL 학습이 끝난 후 다시 데이터셋에 라벨을 포함하여 SL 방식으로 Logistic Regression 모델을 학습시켜 SSL이 잘 학습이 되었는지 확인한다.
5. Grad-CAM을 활용하여 모델이 어떤 부분을 보고 예측하는지 시각화하여 분석한다.


## ⚙️ 데이터셋
- **AI Hub 주요질환 이미지 합성데이터셋(X-ray)**
- **MURA**

  
## 🔎 전처리 
- **CLAHE** : 이미지 대비 개선을 위해 사용

| 적용 전 | 적용 후 |
|----------|----------|
| <img src="https://github.com/user-attachments/assets/0078b445-94fe-455a-83b3-3ccc271b987b" height="500"> | <img src="https://github.com/user-attachments/assets/aa9c8be1-c263-4bc4-84f2-d657ab351afc" height="500"> |


## 💻 사용 모델 설명
### SimCLR
<img src="https://github.com/user-attachments/assets/f5142624-5940-425f-8000-d558b07b201c" width="700">

### BYOL
<img src="https://github.com/user-attachments/assets/7048f947-e238-4914-8ee9-23b661b3c4e9" width="700">

### GradCAM  
<img src="https://github.com/user-attachments/assets/71349624-31b6-4cd2-8796-1c39adaa0459" width="700">


## 🖼️ 사용한 데이터 증강 기법
| **Random GaussianBlur** | **Random Flip** | **Random Rotation** | **Chess board** | **Random Erasing** |
|--------------------------|------------------|---------------------|------------------|--------------------|
| ![image](https://github.com/user-attachments/assets/75251027-3165-4b0c-b817-185c392092c9) | ![image](https://github.com/user-attachments/assets/67c0dbc2-d2e5-43ad-910e-43c5b3017752) | ![image](https://github.com/user-attachments/assets/a2e60c88-f1e1-465d-825f-ee818eca579d) | ![image](https://github.com/user-attachments/assets/4c68f9a8-46ee-490c-b07d-26db77415ae4) | ![image](https://github.com/user-attachments/assets/a1a53277-7357-43b8-9673-4a769b8c25b1) |

## 🖼️ 사용한 데이터 증강 기법
| **Random GaussianBlur** | **Random Flip** | **Random Rotation** | **Chess board** | **Random Erasing** |
|--------------------------|------------------|---------------------|------------------|--------------------|
| ![image](https://github.com/user-attachments/assets/75251027-3165-4b0c-b817-185c392092c9) | ![image](https://github.com/user-attachments/assets/67c0dbc2-d2e5-43ad-910e-43c5b3017752) | ![image](https://github.com/user-attachments/assets/a2e60c88-f1e1-465d-825f-ee818eca579d) | ![image](https://github.com/user-attachments/assets/4c68f9a8-46ee-490c-b07d-26db77415ae4) | ![image](https://github.com/user-attachments/assets/a1a53277-7357-43b8-9673-4a769b8c25b1) |
| <img src="https://github.com/user-attachments/assets/75251027-3165-4b0c-b817-185c392092c9" width="500" /> | <img src="https://github.com/user-attachments/assets/67c0dbc2-d2e5-43ad-910e-43c5b3017752" width="500" /> | <img src="https://github.com/user-attachments/assets/a2e60c88-f1e1-465d-825f-ee818eca579d" width="500" /> | <img src="https://github.com/user-attachments/assets/4c68f9a8-46ee-490c-b07d-26db77415ae4" width="500" /> | <img src="https://github.com/user-attachments/assets/a1a53277-7357-43b8-9673-4a769b8c25b1" width="500" /> |


## 📈 실험 결과
<img src="https://github.com/user-attachments/assets/fd5a9c07-3597-4843-82e3-894be8a7d9bd" width="700">

### SimCLR
<img src="https://github.com/user-attachments/assets/a55cb1cf-9187-40e7-a382-f153d06f4997" width="700">

### BYOL
<img src="https://github.com/user-attachments/assets/34320171-b3fa-420a-bc82-2b870e3fb6f4" width="700">

### SL
<img src="https://github.com/user-attachments/assets/38f9cd08-0f28-4805-bb9e-1d86f7c4969c" width="700">

### GradCAM
<img src="https://github.com/user-attachments/assets/d6f8aac8-bb8b-4cb0-8098-c99c2bfd5e79" width="700">

## 🛠️ **설치 및 실행 방법**
1. **Clone the repository**
   ```bash
   git clone https://github.com/kwminji/Medical-SSL-Augmentation-Research.git
   cd Medical-SSL-Augmentation-Research
   ```
   
2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```
   
3. **Run**
   ```bash
   ### SimCLR
   cd SimCLR
   python train_epoch.py
   python evaluate.py
   
   ### BYOL
   cd BYOL
   python trainer.py
   python evaluate.py
   
   ### SL 
   cd SL
   python train.py
   python evaluate.py
   ```

### 👏 결론
- **SimCLR**
  - 데이터 불균형에 의한 성능 저하
  - 의료 데이터 특성 상 Negative Pair 선정 어려움이 존재
  - Data Augmentation 기법에 예민함

- **BYOL**
  - 데이터 불균형에 강함
  - 비교적 Data Augmentation에 예민하지 않음

