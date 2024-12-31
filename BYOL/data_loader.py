import os
import json
from sklearn.model_selection import train_test_split

def load_ahub_data(base_dir):
    """
    AIHub 데이터를 로드합니다.
    """
    all_data = []
    label_dir = os.path.join(base_dir, "02.라벨링데이터")
    print(f"{label_dir}에서 AIHub 데이터를 로드합니다...")

    for category in os.listdir(label_dir):
        category_path = os.path.join(label_dir, category)
        if os.path.isdir(category_path):
            print(f"카테고리 처리 중: {category}")
            for file_name in os.listdir(category_path):
                if file_name.endswith(".json"):
                    json_path = os.path.join(category_path, file_name)
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        image_path = os.path.join(data["filepath"], data["filename"])
                        label = 1 if data["label"] == "Normal" else 0  # Normal → 1, Abnormal → 0
                        all_data.append({"image_path": image_path, "label": label})

    print(f"AIHub 데이터 총 샘플 수: {len(all_data)}개")
    return all_data

def load_mura_data(mura_base_dir):
    """
    MURA 데이터를 로드합니다.
    """
    all_data = []
    print(f"{mura_base_dir}에서 MURA 데이터를 로드합니다...")

    for root, _, files in os.walk(mura_base_dir):
        for file in files:
            if file.endswith(".png"):  # 이미지 파일만 처리
                label = 1 if "positive" in root else 0  # Positive → 1, Negative → 0
                image_path = os.path.join(root, file)
                all_data.append({"image_path": image_path, "label": label})

    print(f"MURA 데이터 총 샘플 수: {len(all_data)}개")
    return all_data

def load_and_split_data(ahub_base_dir, mura_base_dir, test_size=0.2, random_state=42):
    """
    AIHub 및 MURA 데이터를 통합하여 학습용과 평가용으로 분리합니다.
    """
    ahub_data = load_ahub_data(ahub_base_dir)
    mura_data = load_mura_data(mura_base_dir)
    
    # 데이터 통합
    all_data = ahub_data + mura_data
    print(f"총 데이터 샘플 수 (통합): {len(all_data)}개")

    # 데이터 분리
    train_data, test_data = train_test_split(all_data, test_size=test_size, random_state=random_state)
    print(f"학습 데이터 크기: {len(train_data)}개")
    print(f"평가 데이터 크기: {len(test_data)}개")

    return train_data, test_data
