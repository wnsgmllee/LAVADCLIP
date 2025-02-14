import os
import cv2
from tqdm import tqdm

def extract_frames(video_path, output_dir):
    """비디오에서 프레임을 추출하여 지정된 디렉토리에 저장 (JPG, 품질 98%)"""
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with tqdm(total=total_frames, desc=f"Extracting {os.path.basename(video_path)}") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            frame_filename = os.path.join(output_dir, f"{frame_count}.jpg")
            cv2.imwrite(frame_filename, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 98])
            pbar.update(1)

    cap.release()
    print(f"Extracted {frame_count} frames from {video_path}")

def process_ucf_crime_dataset(root_dir, output_root):
    """UCF-Crime 데이터셋의 모든 비디오에 대해 프레임을 추출 (JPG 저장)"""
    for dirpath, _, filenames in os.walk(root_dir):
        # 모든 .mp4 파일을 찾음
        videos = [f for f in filenames if f.endswith(".mp4")]

        for video in videos:
            video_path = os.path.join(dirpath, video)

            # `dirpath`의 루트부터의 경로 구조를 유지하기 위해 상대 경로 계산
            relative_path = os.path.relpath(dirpath, root_dir)
            output_dir = os.path.join(output_root, relative_path, os.path.splitext(video)[0])

            extract_frames(video_path, output_dir)

if __name__ == "__main__":
    ucf_crime_path = "../Data/ucf"
    output_path = "../Data/ucf_frame"
    process_ucf_crime_dataset(ucf_crime_path, output_path)

