import os
import sys
import yaml

# Thiết lập encoding UTF-8 an toàn cho console Windows
if sys.platform.startswith("win"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
YAML_PATH = os.path.join(CURRENT_DIR, "config_2_datasets.yaml")

def load_config():
    """Đọc file YAML cấu hình 2 dataset."""
    with open(YAML_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# Nạp sẵn dictionary cấu hình
CONFIG = load_config()
GLOBAL_SETTINGS = CONFIG["global_settings"]
ELECTRICITY_CONFIG = CONFIG["datasets"]["electricity"]
TRAFFIC_CONFIG = CONFIG["datasets"]["traffic"]

def get_dataset_hyperparameters(dataset_name: str) -> dict:
    """
    Lấy bộ siêu tham số cho dataset ('electricity' hoặc 'traffic').
    """
    ds = dataset_name.lower()
    if ds not in CONFIG["datasets"]:
        raise ValueError(f"Dataset '{dataset_name}' không hợp lệ. Chọn 'electricity' hoặc 'traffic'.")
    return CONFIG["datasets"][ds]

if __name__ == "__main__":
    print("=" * 70)
    print(" KIỂM TRA FILE CẤU HÌNH 2 DATASET (ELECTRICITY & TRAFFIC)")
    print("=" * 70)
    for name in ["electricity", "traffic"]:
        data = get_dataset_hyperparameters(name)
        params = data["model_params"]
        bench = data["benchmark_paper"]
        print(f"\n[*] DATASET: {name.upper()}")
        print(f" - Mô tả           : {data['description']}")
        print(f" - Số chuỗi        : {data['num_series']} time-series | {data['total_time_points']:,} time-points")
        print(f" - Cấu hình mạng   : hidden={params['hidden_size']}, n_enc={params['num_encoder_layers']}, n_dec={params['num_decoder_layers']}, p={params['decoder_output_dim']}")
        print(f" - Kỹ thuật        : revIn={params['rev_in']}, layerNorm={params['layer_norm']}, dropout={params['dropout_level']}")
        print(f" - Kết quả Paper   : Avg MSE = {bench['average']['mse']} | Avg MAE = {bench['average']['mae']}")
    print("\n" + "=" * 70)
    print(" [V] TẤT CẢ THÔNG SỐ ĐÃ ĐƯỢC NẠP CHÍNH XÁC TỪ FILE YAML!")
    print("=" * 70)
