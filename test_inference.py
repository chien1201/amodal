import torch
from PIL import Image
from torchvision.transforms import ToTensor
from efficient_sam.build_efficient_sam import build_efficient_sam_vits

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. 載入模型結構與權重
model = build_efficient_sam_vits()
checkpoint = torch.load("weights/efficient_sam_vits.pt", map_location=device)

if "model" in checkpoint:
    state_dict = checkpoint["model"]
else:
    state_dict = checkpoint

model.load_state_dict(state_dict)
model.to(device).eval()

# 2. 構造測試輸入張量
# 影像形狀: [Batch, Channels, Height, Width]
dummy_img = torch.randn(1, 3, 1024, 1024).to(device)

# 3. 構造 Bounding Box 及其對應的 Labels
# Bounding Box 在 EfficientSAM 中通常表示為 2 個角點：[x1, y1] 與 [x2, y2]
# 形狀為: [Batch, Num_queries, Num_points_per_query, 2] -> [1, 1, 2, 2]
dummy_coords = (
    torch.tensor([[[[100.0, 100.0], [300.0, 300.0]]]]).float().to(device)
)

# 依據 SAM 定義：Box 的左上角標籤為 2，右下角標籤為 3
# 形狀為: [Batch, Num_queries, Num_points_per_query] -> [1, 1, 2]
dummy_labels = torch.tensor([[[2, 3]]]).to(device)

# 4. 執行推論
with torch.no_grad():
    # 同時傳入 coords 與 labels
    predicted_logits, predicted_iou = model(
        dummy_img,
        dummy_coords,
        dummy_labels,
    )
    print("Inference 成功！")
    print(f"Mask shape: {predicted_logits.shape}")
    print(f"Predicted IoU: {predicted_iou}")