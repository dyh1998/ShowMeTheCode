import torch

# 获取cuda的脚本
while True:
    try:
        if torch.cuda.device_count() >= 1:
            print("gpu getting.")
            os.system("nohup python3 text_cls_model.py")
            break
    except Exception as e:
        print("error: ", e)
        print("gpu getting failed.")

# 检查cuda是否可用
if torch.cuda.is_available():
    print("cuda got.")
