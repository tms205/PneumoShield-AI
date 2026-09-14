import http.server
import socketserver
import json
import urllib.parse
import io
import re
import os
import shutil
import time
import sys
import math
from pathlib import Path
from PIL import Image

# Reconfigure stdout to use UTF-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
import torchvision
from torchvision import transforms

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN & THIẾT BỊ
# ==========================================
WORKSPACE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_RUNS = {
    "scratch": {
        "label": "Train from scratch",
        "dir": WORKSPACE_DIR / "outputs_scratch",
    },
}
OUTPUTS_DIR = OUTPUT_RUNS["scratch"]["dir"]
DATASET_CLEAN_DIR = WORKSPACE_DIR / "dataset_clean_70_15_15"
STATIC_DIR = WORKSPACE_DIR / "web_app" / "static"
SAMPLES_DIR = STATIC_DIR / "samples"

# Thiết bị chạy mô hình
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[System] Using device: {DEVICE}")

# ==========================================
# ĐỊNH NGHĨA BIẾN THƯỜNG TRỰC & PIPELINE
# ==========================================
CLASSES = ("NORMAL", "PNEUMONIA")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def pad_to_square(image):
    width, height = image.size
    side = max(width, height)
    left = (side - width) // 2
    top = (side - height) // 2
    return transforms.functional.pad(
        image,
        [left, top, side - width - left, side - height - top],
        fill=0,
    )


def build_transform(image_size, preprocess_version):
    common_end = [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    if preprocess_version == "center_crop_v1":
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(image_size, antialias=True),
            transforms.CenterCrop((image_size, image_size)),
            *common_end,
        ])
    if preprocess_version == "square_pad_v1":
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Lambda(pad_to_square),
            transforms.Resize((image_size, image_size), antialias=True),
            *common_end,
        ])
    raise ValueError(f"Unsupported preprocessing version: {preprocess_version!r}.")


def checkpoint_uses_current_dataset(checkpoint):
    checkpoint_data_dir = checkpoint.get("data_dir")
    if not checkpoint_data_dir:
        return False
    return Path(str(checkpoint_data_dir)).name == DATASET_CLEAN_DIR.name

# ==========================================
# TẢI CÁC MÔ HÌNH PRE-TRAINED
# ==========================================
model_eff = None
model_vit = None
eff_transform = None
vit_transform = None

def load_models():
    global model_eff, model_vit, eff_transform, vit_transform
    load_errors = []
    
    # 1. Tải EfficientNet-B4
    eff_path = OUTPUTS_DIR / "efficientnet_b4" / "best_efficientnet_b4.pth"
    print(f"[Model] Loading EfficientNet-B4 from: {eff_path} ...")
    try:
        candidate_eff = torchvision.models.efficientnet_b4()
        candidate_eff.classifier[1] = torch.nn.Linear(candidate_eff.classifier[1].in_features, len(CLASSES))
        # Checkpoints are generated locally by this project's training notebooks.
        checkpoint_eff = torch.load(eff_path, map_location=DEVICE, weights_only=False)
        if checkpoint_eff.get("class_names") != list(CLASSES):
            raise ValueError("Checkpoint classes do not match application classes.")
        if not checkpoint_uses_current_dataset(checkpoint_eff):
            raise ValueError("Checkpoint dataset does not match the web app dataset.")
        eff_transform = build_transform(380, checkpoint_eff.get("preprocess_version"))
        candidate_eff.load_state_dict(checkpoint_eff["model_state_dict"])
        model_eff = candidate_eff.to(DEVICE)
        model_eff.eval()
        print(f"[Model] EfficientNet-B4 loaded successfully ({checkpoint_eff.get('preprocess_version')}).")
    except Exception as e:
        model_eff = None
        eff_transform = None
        load_errors.append(f"EfficientNet-B4: {e}")
        print(f"[Error] Failed to load EfficientNet-B4: {e}")

    # 2. Tải ViT-B/16
    vit_path = OUTPUTS_DIR / "vit_b_16" / "best_vit_b_16.pth"
    print(f"[Model] Loading ViT-B/16 from: {vit_path} ...")
    try:
        candidate_vit = torchvision.models.vit_b_16()
        candidate_vit.heads.head = torch.nn.Linear(candidate_vit.heads.head.in_features, len(CLASSES))
        checkpoint_vit = torch.load(vit_path, map_location=DEVICE, weights_only=False)
        if checkpoint_vit.get("class_names") != list(CLASSES):
            raise ValueError("Checkpoint classes do not match application classes.")
        if not checkpoint_uses_current_dataset(checkpoint_vit):
            raise ValueError("Checkpoint dataset does not match the web app dataset.")
        vit_transform = build_transform(224, checkpoint_vit.get("preprocess_version"))
        candidate_vit.load_state_dict(checkpoint_vit["model_state_dict"])
        model_vit = candidate_vit.to(DEVICE)
        model_vit.eval()
        print(f"[Model] ViT-B/16 loaded successfully ({checkpoint_vit.get('preprocess_version')}).")
    except Exception as e:
        model_vit = None
        vit_transform = None
        load_errors.append(f"ViT-B/16: {e}")
        print(f"[Error] Failed to load ViT-B/16: {e}")

    if load_errors:
        raise RuntimeError("Unable to load trained models: " + " | ".join(load_errors))

# ==========================================
# KHỞI TẠO MẪU ẢNH CHO FRONTEND
# ==========================================
def prepare_sample_images():
    """Sao chép một số ảnh mẫu từ test set sang thư mục static để phục vụ frontend."""
    # Nếu không có dataset gốc nhưng thư mục samples đã có sẵn ảnh mẫu, tái sử dụng các ảnh này
    if not (DATASET_CLEAN_DIR / "test").exists() and SAMPLES_DIR.exists():
        existing_samples = []
        for file_path in sorted(SAMPLES_DIR.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                label = "PNEUMONIA" if "PNEUMONIA" in file_path.name.upper() else "NORMAL"
                existing_samples.append({
                    "filename": file_path.name,
                    "label": label,
                    "original_name": file_path.name
                })
        if existing_samples:
            print(f"[System] Reusing {len(existing_samples)} pre-packaged sample images in {SAMPLES_DIR.name}.")
            return existing_samples

    if SAMPLES_DIR.exists():
        shutil.rmtree(SAMPLES_DIR)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    
    copied_samples = []
    
    for class_name in CLASSES:
        class_src_dir = DATASET_CLEAN_DIR / "test" / class_name
        if not class_src_dir.exists():
            continue
            
        # Lấy 4 file ảnh đầu tiên của mỗi lớp
        files = sorted(list(class_src_dir.glob("*.jpeg"))) + sorted(list(class_src_dir.glob("*.jpg")))
        for f in files[:4]:
            dest_name = f"{class_name}_{f.name}"
            dest_path = SAMPLES_DIR / dest_name
            if not dest_path.exists():
                shutil.copy2(f, dest_path)
            copied_samples.append({
                "filename": dest_name,
                "label": class_name,
                "original_name": f.name
            })
            
    print(f"[System] Prepared {len(copied_samples)} sample images for quick-diagnosis.")
    return copied_samples


def list_training_outputs():
    """Return PNG/JPG result assets from the scratch output folder."""
    runs = []
    for run_key, run_info in OUTPUT_RUNS.items():
        run_dir = run_info["dir"]
        run_payload = {
            "key": run_key,
            "label": run_info["label"],
            "exists": run_dir.exists(),
            "models": [],
        }

        if run_dir.exists():
            for model_dir in sorted([p for p in run_dir.iterdir() if p.is_dir()]):
                images = []
                for image_path in sorted(model_dir.iterdir()):
                    if image_path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                        continue
                    images.append({
                        "filename": image_path.name,
                        "title": image_path.stem.replace("_", " ").title(),
                        "url": (
                            "/api/training-output-image?"
                            + urllib.parse.urlencode({
                                "run": run_key,
                                "model": model_dir.name,
                                "image": image_path.name,
                            })
                        ),
                    })
                if images:
                    run_payload["models"].append({
                        "key": model_dir.name,
                        "label": model_dir.name.replace("_", " ").title(),
                        "images": images,
                    })

        runs.append(run_payload)
    return runs


def resolve_training_output_image(run_key, model_name, image_name):
    if run_key not in OUTPUT_RUNS:
        return None
    if Path(model_name).name != model_name or Path(image_name).name != image_name:
        return None
    image_path = OUTPUT_RUNS[run_key]["dir"] / model_name / image_name
    if not image_path.is_file() or image_path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
        return None
    try:
        image_path.resolve().relative_to(OUTPUT_RUNS[run_key]["dir"].resolve())
    except ValueError:
        return None
    return image_path

# ==========================================
# THỰC HIỆN DỰ ĐOÁN
# ==========================================
def predict_image(image_bytes, w_eff=0.5, w_vit=0.5):
    """
    Tiền xử lý ảnh và chạy suy luận thông qua các mô hình.
    w_eff, w_vit: Trọng số của từng mô hình (tổng = 1.0)
    """
    if model_eff is None or model_vit is None:
        raise RuntimeError("Trained models have not been loaded successfully.")

    try:
        w_eff = float(w_eff)
        w_vit = float(w_vit)
    except (TypeError, ValueError):
        raise ValueError("Model weights must be numeric.")
    if not math.isfinite(w_eff) or not math.isfinite(w_vit) or w_eff < 0 or w_vit < 0:
        raise ValueError("Model weights must be finite non-negative values.")
    weight_sum = w_eff + w_vit
    if weight_sum <= 0:
        raise ValueError("At least one model weight must be positive.")
    w_eff /= weight_sum
    w_vit /= weight_sum

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Định dạng ảnh không hợp lệ: {e}")
        
    start_time = time.time()
    
    # 1. Chạy EfficientNet-B4
    time_eff = 0.0
    t0 = time.time()
    with torch.no_grad():
        img_eff = eff_transform(img).unsqueeze(0).to(DEVICE)
        logits_eff = model_eff(img_eff)
        prob_eff = torch.softmax(logits_eff, dim=1).squeeze().cpu().tolist()
    time_eff = time.time() - t0

    # 2. Chạy ViT-B/16
    time_vit = 0.0
    t0 = time.time()
    with torch.no_grad():
        img_vit = vit_transform(img).unsqueeze(0).to(DEVICE)
        logits_vit = model_vit(img_vit)
        prob_vit = torch.softmax(logits_vit, dim=1).squeeze().cpu().tolist()
    time_vit = time.time() - t0
        
    # 3. Tính toán Ensemble (Trung bình trọng số)
    # prob = [NORMAL, PNEUMONIA]
    prob_ens = [
        w_eff * prob_eff[0] + w_vit * prob_vit[0],
        w_eff * prob_eff[1] + w_vit * prob_vit[1]
    ]
    
    # Chuẩn hóa lại tổng xác suất ensemble bằng 1.0 đề phòng sai số
    sum_ens = sum(prob_ens)
    prob_ens = [p / sum_ens for p in prob_ens]
    
    # Quyết định nhãn cuối cùng
    pred_idx = 1 if prob_ens[1] >= 0.5 else 0
    pred_label = CLASSES[pred_idx]
    confidence = prob_ens[pred_idx]
    
    total_time = time.time() - start_time
    
    return {
        "success": True,
        "prediction": pred_label,
        "confidence": confidence,
        "time_taken_sec": total_time,
        "efficientnet": {
            "normal": prob_eff[0],
            "pneumonia": prob_eff[1],
            "time_sec": time_eff
        },
        "vit": {
            "normal": prob_vit[0],
            "pneumonia": prob_vit[1],
            "time_sec": time_vit
        },
        "ensemble": {
            "normal": prob_ens[0],
            "pneumonia": prob_ens[1],
            "w_eff": w_eff,
            "w_vit": w_vit
        }
    }

# ==========================================
# CUSTOM REQUEST HANDLER
# ==========================================
class DiagnosticsRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Thiết lập directory để phục vụ file tĩnh mặc định
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def end_headers(self):
        # CORS headers cho phát triển local
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        request_path = parsed_url.path

        # 1. API lấy danh sách ảnh mẫu
        if request_path == "/api/samples":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            
            samples = []
            if SAMPLES_DIR.exists():
                for f in SAMPLES_DIR.glob("*"):
                    if f.suffix.lower() in (".jpeg", ".jpg", ".png"):
                        # Trích xuất nhãn từ tên file (NORMAL_... hoặc PNEUMONIA_...)
                        label = "NORMAL" if f.name.startswith("NORMAL_") else "PNEUMONIA"
                        samples.append({
                            "filename": f.name,
                            "label": label,
                            "url": f"/static/samples/{f.name}"
                        })
            
            self.wfile.write(json.dumps(samples).encode("utf-8"))
            return

        if request_path == "/api/training-outputs":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(list_training_outputs()).encode("utf-8"))
            return

        if request_path == "/api/training-output-image":
            query_params = urllib.parse.parse_qs(parsed_url.query)
            run_key = query_params.get("run", [""])[0]
            model_name = query_params.get("model", [""])[0]
            image_name = query_params.get("image", [""])[0]
            image_path = resolve_training_output_image(run_key, model_name, image_name)

            if image_path is None:
                self.send_error(404, "Training output image not found")
                return

            self.send_response(200)
            if image_path.suffix.lower() in (".jpg", ".jpeg"):
                self.send_header("Content-Type", "image/jpeg")
            else:
                self.send_header("Content-Type", "image/png")
            self.end_headers()

            with open(image_path, "rb") as f:
                self.wfile.write(f.read())
            return
            
        # 2. Phục vụ thư mục static/samples
        elif request_path.startswith("/static/"):
            # Chuyển đổi URL tĩnh thành đường dẫn tệp thực
            relative_path = request_path[8:] # Bỏ "/static/"
            file_path = STATIC_DIR / relative_path
            
            if file_path.exists() and file_path.is_file():
                self.send_response(200)
                # Thiết lập content type
                if file_path.suffix == ".html":
                    self.send_header("Content-Type", "text/html")
                elif file_path.suffix == ".css":
                    self.send_header("Content-Type", "text/css")
                elif file_path.suffix == ".js":
                    self.send_header("Content-Type", "application/javascript")
                elif file_path.suffix in (".jpg", ".jpeg"):
                    self.send_header("Content-Type", "image/jpeg")
                elif file_path.suffix == ".png":
                    self.send_header("Content-Type", "image/png")
                self.end_headers()
                
                with open(file_path, "rb") as f:
                    self.wfile.write(f.read())
                return
            else:
                self.send_error(404, "File not found")
                return

        # Mặc định gọi hàm cha để phục vụ static files (như index.html)
        if request_path == "/" or request_path == "":
            self.path = "/index.html"
            
        super().do_GET()

    def do_POST(self):
        # API Dự đoán ảnh
        if self.path.startswith("/api/predict"):
            # Lấy các tham số query (để lấy trọng số w_eff, w_vit hoặc tên ảnh mẫu)
            parsed_url = urllib.parse.urlparse(self.path)
            query_params = urllib.parse.parse_qs(parsed_url.query)
            
            w_eff = query_params.get("w_eff", [0.5])[0]
            w_vit = query_params.get("w_vit", [0.5])[0]
            sample_file = query_params.get("sample", [None])[0]
            
            image_bytes = None
            
            # Trường hợp 1: Dự đoán ảnh mẫu có sẵn trên server
            if sample_file:
                if Path(sample_file).name != sample_file:
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"success": False, "error": "Invalid sample filename."}).encode("utf-8"))
                    return
                target_path = SAMPLES_DIR / sample_file
                if target_path.exists() and target_path.is_file():
                    with open(target_path, "rb") as f:
                        image_bytes = f.read()
                else:
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"success": False, "error": f"Không tìm thấy tệp ảnh mẫu: {sample_file}"}).encode("utf-8"))
                    return
            
            # Trường hợp 2: Upload file ảnh trực tiếp
            else:
                content_type = self.headers.get("Content-Type", "")
                content_length = int(self.headers.get("Content-Length", 0))
                
                if "multipart/form-data" in content_type:
                    boundary_parts = content_type.split("boundary=")
                    if len(boundary_parts) > 1:
                        boundary = ("--" + boundary_parts[1]).encode("utf-8")
                        raw_data = self.rfile.read(content_length)
                        
                        parts = raw_data.split(boundary)
                        for part in parts:
                            if b"filename=" in part:
                                header_end = part.find(b"\r\n\r\n")
                                if header_end != -1:
                                    file_content = part[header_end + 4:]
                                    if file_content.endswith(b"\r\n"):
                                        file_content = file_content[:-2]
                                    if file_content.endswith(b"--\r\n"):
                                        file_content = file_content[:-4]
                                    if file_content.endswith(b"--"):
                                        file_content = file_content[:-2]
                                    
                                    image_bytes = file_content
                                    break
            
            if image_bytes is None:
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": "Không nhận được dữ liệu ảnh"}).encode("utf-8"))
                return
                
            # Thực hiện chẩn đoán
            try:
                result = predict_image(image_bytes, w_eff=w_eff, w_vit=w_vit)
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps(result).encode("utf-8"))
            except ValueError as e:
                self.send_response(400)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode("utf-8"))
            return

        self.send_error(404, "Endpoint not found")

# ==========================================
# KHỞI CHẠY SERVER
# ==========================================
def main():
    # 1. Nạp mô hình PyTorch
    load_models()
    
    # 2. Tạo ảnh mẫu cho frontend
    prepare_sample_images()
    
    PORT = int(os.environ.get("PORT", "8000"))
    # Cho phép tái sử dụng địa chỉ cổng nhanh chóng
    socketserver.TCPServer.allow_reuse_address = True
    
    with socketserver.TCPServer(("", PORT), DiagnosticsRequestHandler) as httpd:
        print("\n" + "="*60)
        print(f"  [Server] Diagnostics server running at:")
        print(f"           >>> http://localhost:{PORT} <<<")
        print("  [System] Press Ctrl+C to stop the server.")
        print("="*60 + "\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[System] Stopping server...")
            httpd.server_close()
            print("[System] Server stopped successfully.")

if __name__ == "__main__":
    main()
