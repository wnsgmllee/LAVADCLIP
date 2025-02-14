import torch
import clip
import os
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline, Blip2ForConditionalGeneration, Blip2Processor
from ucf_option import args
from dotenv import load_dotenv


# 📌 환경 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model_path = os.path.join(args.model_save_path, args.backbone + ".pt")
load_dotenv()
HF_TOKEN = os.getenv("HUGGING_FACE_TOKEN")

# 📌 CLIP 모델 로드
model, preprocess = clip.load(args.backbone, device=device, download_root=clip_model_path)

# 📌 BLIP2 모델 로드
blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-6.7b-coco")
blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-6.7b-coco").to(device)

# 📌 Llama-2 모델 설정
llama_model_name = "meta-llama/Llama-2-13b-chat"
tokenizer = AutoTokenizer.from_pretrained(llama_model_name, use_auth_token=HF_TOKEN)
text_generator = pipeline("text-generation", model=llama_model_name, tokenizer=tokenizer, torch_dtype=torch.float16, device_map="auto")

# 📌 LLM 응답 생성 함수
def generate(x, max_length=100):
    sequences = text_generator(x, do_sample=True, top_k=10, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id, max_length=max_length)
    return sequences[0]["generated_text"].replace(x, "").strip()

# 📌 프레임 로드 함수
def load_frames(frame_folder, target_size=(224, 224)):
    frame_list = sorted([os.path.join(frame_folder, f) for f in os.listdir(frame_folder) if f.endswith(".jpg")])
    frames = [cv2.resize(cv2.imread(frame), target_size) for frame in frame_list]
    return np.array(frames)

# 📌 캡션 생성 함수
def generate_caption(frames):
    images = [Image.fromarray(frame) for frame in frames]
    inputs = blip_processor(images=images, return_tensors="pt", padding=True).to(device)
    generated_ids = blip_model.generate(**inputs)
    captions = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)
    return captions

# 📌 LLM을 활용한 이상 현상 키워드 추출 함수
def extract_anomalous_keywords(caption, k):
    prompt = f"Given the following scene description, extract {k} key words that describe potential abnormal activities.\nDescription: {caption}\nKeywords:"
    output_en = generate(prompt, 100)
    return output_en.split()[:k]

# 📌 Feature Extraction (Segment 단위, Batch 적용, padding/drop 지원)
def extract_features_from_frames(frame_folder, save_dir_visual, save_dir_text, segment_size=16, k=5, batch_size=4):
    frames = load_frames(frame_folder)
    total_frames = len(frames)

    if total_frames == 0:
        print(f"⚠️ Skipping {frame_folder} (No Frames)")
        return

    num_segments = total_frames // segment_size
    if num_segments == 0:
        print(f"⚠️ Skipping {frame_folder} (Too Short)")
        return

    vision_features, text_features = [], []
    
    for i in range(0, len(frames), segment_size):
        segment_frames = frames[i:i + segment_size]

        # 📌 마지막 segment가 16보다 작을 경우 처리
        if len(segment_frames) < segment_size:
            if args.segment_handling == "padding":
                pad_frames = np.tile(segment_frames[-1], (segment_size - len(segment_frames), 1, 1, 1))
                segment_frames = np.concatenate((segment_frames, pad_frames), axis=0)
            elif args.segment_handling == "drop":
                continue  # 16보다 작으면 스킵

        # 📌 Batch 단위로 CLIP 모델 처리
        vision_features_list = []
        for j in range(0, len(segment_frames), batch_size):
            batch = segment_frames[j:j+batch_size]
            batch_tensors = torch.stack([preprocess(Image.fromarray(frame)).to(device) for frame in batch]).to(device)

            with torch.no_grad():
                batch_features = model.encode_image(batch_tensors).cpu().numpy()

            vision_features_list.append(batch_features)

        vision_feature = np.mean(np.vstack(vision_features_list), axis=0)

        # 📌 BLIP2 기반 Caption 생성
        raw_captions = generate_caption(segment_frames)
        anomalous_keywords = extract_anomalous_keywords(" ".join(raw_captions), k)

        # 📌 CLIP Text Feature Encoding
        text_tokens = clip.tokenize([" ".join(anomalous_keywords)]).to(device)
        with torch.no_grad():
            text_feature = model.encode_text(text_tokens).squeeze(0).cpu().numpy()

        vision_features.append(vision_feature)
        text_features.append(text_feature)

        torch.cuda.empty_cache()

    # 📌 Feature 저장
    video_base = os.path.basename(frame_folder)
    save_path_visual = os.path.join(save_dir_visual, f"{video_base}_visual_feature.npy")
    save_path_text = os.path.join(save_dir_text, f"{video_base}_text_feature.npy")

    np.save(save_path_visual, np.array(vision_features))
    np.save(save_path_text, np.array(text_features))
    print(f"✅ Feature extraction completed for {frame_folder}")

# 📌 메인 실행 코드
if __name__ == '__main__':
    frame_root = args.frame_path
    for category in tqdm(os.listdir(frame_root), desc="Processing Categories"):
        category_path = os.path.join(frame_root, category)
        save_category_visual = os.path.join(args.feature_LLM_save_path, args.backbone.replace("/", "-"), "visual_features", category)
        save_category_text = os.path.join(args.feature_LLM_save_path, args.backbone.replace("/", "-"), "text_features", category)
        os.makedirs(save_category_visual, exist_ok=True)
        os.makedirs(save_category_text, exist_ok=True)
        
        for video_name in os.listdir(category_path):
            frame_folder = os.path.join(category_path, video_name)
            video_base = os.path.basename(frame_folder)
            
            # 📌 이미 처리된 경우 스킵
            if os.path.exists(os.path.join(save_category_visual, f"{video_base}_visual_feature.npy")):
                print(f"⏩ Skipping {video_name} (Already Processed)")
                continue
            
            print(f"✅ Processing Frames for: {video_name}")
            extract_features_from_frames(frame_folder, save_category_visual, save_category_text)

