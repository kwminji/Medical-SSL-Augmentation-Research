import os
import json

def load_ahub_data(base_dir):
    """
    AIHub 데이터를 로드합니다.
    """
    all_data = []
    print(f"{base_dir}에서 AIHub 데이터를 로드합니다...")

    for category in os.listdir(base_dir):
        category_path = os.path.join(base_dir, category)
        if os.path.isdir(category_path):
            print(f"카테고리 처리 중: {category}")
            for file_name in os.listdir(category_path):
                if file_name.endswith(".json"):
                    json_path = os.path.join(category_path, file_name)
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        image_path = os.path.join(data["filepath"], data["filename"])
                        label = 1 if "Normal" in category else 0  # Normal → 1, Abnormal → 0
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

def load_and_split_data(train_ahub_dir, train_mura_dir, valid_ahub_dir=None, valid_mura_dir=None):
    """
    학습 데이터와 검증 데이터를 분리하여 로드합니다.
    """
    print("학습 데이터 로드 중...")
    train_ahub_data = load_ahub_data(train_ahub_dir)
    train_mura_data = load_mura_data(train_mura_dir)
    all_train_data = train_ahub_data + train_mura_data

    validation_data = []
    if valid_ahub_dir and valid_mura_dir:
        print("검증 데이터 로드 중...")
        valid_ahub_data = load_ahub_data(valid_ahub_dir)
        valid_mura_data = load_mura_data(valid_mura_dir)
        validation_data = valid_ahub_data + valid_mura_data

    print(f"학습 데이터 크기: {len(all_train_data)}개")
    if validation_data:
        print(f"검증 데이터 크기: {len(validation_data)}개")
    else:
        print("검증 데이터를 로드하지 않습니다.")

    return all_train_data, validation_data

# 실행 예시
if __name__ == "__main__":
    # 학습 데이터 디렉토리
    train_ahub_dir = "/home/work/VisionAI/CLAHE_train/aihub/02.라벨링데이터"
    train_mura_dir = "/home/work/VisionAI/CLAHE_train/MURA"

    # 검증 데이터 디렉토리
    valid_ahub_dir = "/home/work/VisionAI/CLAHE_valid/aihub/02.라벨링데이터"
    valid_mura_dir = "/home/work/VisionAI/CLAHE_valid/MURA"

    # 데이터 로드 및 분리
    train_data, val_data = load_and_split_data(
        train_ahub_dir, train_mura_dir, valid_ahub_dir, valid_mura_dir
    )
