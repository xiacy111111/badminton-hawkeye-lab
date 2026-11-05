# court_config.py — 创建这个文件
"""羽毛球场地标准尺寸常量 (单位: 米)"""

# 全场
L = 13.40
W_DOUBLES = 6.10
W_SINGLES = 5.18

# 半场
HL = L / 2          # 6.70
HW_D = W_DOUBLES / 2  # 3.05
HW_S = W_SINGLES / 2  # 2.59

# 发球线
SERVICE_FRONT = 1.98   # 前发球线距网
SERVICE_BACK = 0.72    # 双打后发球线距底线

# 网
NET_H = 1.55
NET_POST_SPREAD = 0.5

# 兼容旧名
W = W_DOUBLES
HW = HW_D
COURT_LENGTH = L
COURT_WIDTH = W_DOUBLES