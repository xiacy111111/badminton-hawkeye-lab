
#!/usr/bin/env python3
"""蹇€熸紨绀猴細鐢ㄥ悎鎴愭暟鎹繍琛屼氦鍙夎瘎鍒嗚瀺鍚�"""

import numpy as np
from cross_scoring_fusion import CrossScoringFusion

# 妯℃嫙涓€鏉＄畝鍗曡建杩圭殑瑙傛祴鏁版嵁
print("Good-Badminton 浜ゅ弶璇勫垎铻嶅悎婕旂ず")
print("=" * 50)

# 鐢熸垚鍚堟垚瑙傛祴鏁版嵁锛堢窘姣涚悆鍚庡崐娈甸琛岋級
np.random.seed(42)
g = 9.81

# 鐪熷疄杞ㄨ抗鍙傛暟
true_landing = np.array([2.5, 1.2])  # 鐪熷疄钀界偣

# 妯℃嫙鏈€鍚�0.3绉掔殑瑙傛祴锛堟渶鍚庤娴嬬偣绂诲湴0.5m锛�
n_obs = 15
obs_t = np.linspace(0, 0.3, n_obs)
obs_pts = np.zeros((n_obs, 3))

# 涓€涓畝鍗曠殑鎶涚墿绾夸笅钀�
for i in range(n_obs):
    t = obs_t[i]
    # 妯℃嫙涓€涓湁闃诲姏鐨勪笅钀�
    obs_pts[i, 0] = 2.5 + (t - 0.3) * 5.0  # x鏂瑰悜
    obs_pts[i, 1] = 1.2 + (t - 0.3) * 2.0  # y鏂瑰悜
    obs_pts[i, 2] = 0.5 + (0.3 - t) * 3.0   # z鏂瑰悜

# 鍔犲櫔澹�
obs_pts += np.random.randn(n_obs, 3) * 0.002

print(f"瑙傛祴鐐�: {n_obs}涓�, 鎸佺画{obs_t[-1]:.3f}s")
print(f"鐪熷疄钀界偣: ({true_landing[0]:.3f}, {true_landing[1]:.3f})")
print(f"鏈€鍚庤娴嬬偣: ({obs_pts[-1,0]:.3f}, {obs_pts[-1,1]:.3f}, {obs_pts[-1,2]:.3f})")
print()

# 杩愯浜ゅ弶璇勫垎铻嶅悎
fusion = CrossScoringFusion(obs_pts, obs_t)
result = fusion.fuse()

print(f"棰勬祴钀界偣: ({result['landing_x']:.3f}, {result['landing_y']:.3f})")
print(f"璇樊: {np.linalg.norm([result['landing_x']-true_landing[0], result['landing_y']-true_landing[1]])*1000:.1f}mm")
print(f"缃俊搴�: {result['confidence']:.3f}")
print(f"鑳滃嚭鏂规硶: {result['winning_method']}")
print(f"鑰楁椂: {result['opt_time_ms']:.0f}ms")
print()
print("鎵€鏈夋柟娉曡惤鐐�:")
for name, (x, y) in result['all_landings'].items():
    marker = " 鈫� 鑳滃嚭" if name == result['winning_method'] else ""
    print(f"  {name}: ({x:.3f}, {y:.3f}){marker}")

print()
print("鍚勬柟娉曡瘎鍒�:")
for name, score in result['all_scores'].items():
    print(f"  {name}: {score:.4f}")

print()
print("鉁� 婕旂ず瀹屾垚锛�")