import json
import requests

BASE_URL = "http://127.0.0.1:8000"

# 1. Info endpoint
info_resp = requests.get(f"{BASE_URL}/experimental/dent/info")
print("INFO ENDPOINT:")
print(json.dumps(info_resp.json(), indent=2))

# 2. Dent positive sample: dents103.jpg
pos_path = r"C:\Users\susha\Downloads\CR7-DET\CR7-DET\image\dents103.jpg"
with open(pos_path, "rb") as f:
    resp_pos = requests.post(
        f"{BASE_URL}/experimental/dent/detect",
        params={"confidence_threshold": 0.30},
        files={"image": ("dents103.jpg", f, "image/jpeg")},
    )
print("\nDENT POSITIVE (dents103.jpg) - Threshold 0.30:")
print(json.dumps(resp_pos.json(), indent=2))

# 3. Non-dent sample: inclusions104.jpg
neg_path = r"C:\Users\susha\Downloads\CR7-DET\CR7-DET\image\inclusions104.jpg"
with open(neg_path, "rb") as f:
    resp_neg = requests.post(
        f"{BASE_URL}/experimental/dent/detect",
        params={"confidence_threshold": 0.30},
        files={"image": ("inclusions104.jpg", f, "image/jpeg")},
    )
print("\nNON-DENT SAMPLE (inclusions104.jpg) - Threshold 0.30:")
print(json.dumps(resp_neg.json(), indent=2))
