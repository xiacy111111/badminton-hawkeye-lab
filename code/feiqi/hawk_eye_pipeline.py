
#!/usr/bin/env python3
"""
hawk_eye_pipeline.py 鈥� Good-Badminton 楣扮溂鍒ょ綒Pipeline (v3.0)

鏇存柊锛�
  - 鏀寔涓夌铻嶅悎绛栫暐锛氱畝鍗曞钩鍧� / 浜ゅ弶璇勫垎 / 绮剧‘UMTO
  - 2绉掑疄鏃跺垎鏋愰绠�
  - 瀹屾暣鐨勭粨鏋滆緭鍑哄拰鍙鍖栨暟鎹�

娴佺▼锛�
  瑙嗛杈撳叆 鈫� 鐞冩娴� + 涓夎娴嬮噺 鈫� 6鏂规硶鎻愭
  鈫� [铻嶅悎绛栫暐] 鈫� 鍒ょ綒鍐崇瓥 鈫� 鍙鍖栨覆鏌� 鈫� 缁撴灉杈撳嚭
"""

import numpy as np
import time
import json
from cross_scoring_fusion import CrossScoringFusion
from trajectory_methods import run_all_methods


class HawkEyePipeline:
    """
    瀹屾暣鍒ょ綒Pipeline

    Parameters
    ----------
    calib_params : dict 鈥� 鍙岀洰鏍囧畾鍙傛暟
    physics_params : dict 鈥� 榛樿鐗╃悊鍙傛暟
    fusion_strategy : str 鈥� 'cross_scoring' | 'precise' | 'average' | 'best_single'
    """

    def __init__(self, calib_params=None, physics_params=None, 
                 fusion_strategy='cross_scoring'):
        self.calib = calib_params or {}
        self.physics = physics_params or {
            'mass': 0.005, 'drag_linear': 0.0, 'drag_quad': 0.001, 'g': 9.81
        }
        self.fusion_strategy = fusion_strategy
        self.timings = {}

    def process(self, obs_3d, obs_t, ground_truth=None):
        """
        澶勭悊涓€鏉″凡涓夎娴嬮噺鐨�3D杞ㄨ抗

        Parameters
        ----------
        obs_3d : ndarray (N, 3) 鈥� 3D杞ㄨ抗鐐�
        obs_t : ndarray (N,) 鈥� 鏃堕棿鎴�
        ground_truth : tuple (x, y) or None 鈥� 鐪熷疄钀界偣锛堢敤浜庤瘎浼帮級

        Returns
        -------
        dict 鈥� 鍒ょ綒缁撴灉
        """
        t_total = time.time()
        N = len(obs_3d)

        if N < 4:
            return {'error': f'杞ㄨ抗鐐逛笉瓒�: {N}涓偣锛堥渶瑕�>=4锛�', 'status': 'FAIL'}

        # ============================================
        # Step 1: 杩愯铻嶅悎
        # ============================================
        t1 = time.time()

        if self.fusion_strategy == 'cross_scoring':
            fusion = CrossScoringFusion(obs_3d, obs_t, self.physics)
            result = fusion.fuse()
        elif self.fusion_strategy == 'precise':
            from precise_fusion import PreciseFusion
            pf = PreciseFusion(obs_3d, obs_t, self.physics)
            result = pf.optimize(use_global=True, max_iter_global=80, max_iter_local=150)
        elif self.fusion_strategy == 'average':
            methods = run_all_methods(obs_3d, obs_t, self.physics)
            valid = [m['landing_xy'] for m in methods.values() if 'landing_xy' in m]
            if valid:
                avg_landing = np.mean(valid, axis=0)
                result = {
                    'landing_x': float(avg_landing[0]),
                    'landing_y': float(avg_landing[1]),
                    'landing_t': obs_t[-1] + 0.3,
                    'confidence': 0.5,
                    'winning_method': 'average',
                    'all_landings': {k: tuple(v.get('landing_xy', [np.nan, np.nan])) 
                                    for k, v in methods.items()},
                    'opt_time_ms': 0,
                }
            else:
                return {'error': '鎵€鏈夋柟娉曞け璐�', 'status': 'FAIL'}
        elif self.fusion_strategy == 'best_single':
            # 鐢ㄤ氦鍙夎瘎鍒嗚嚜鍔ㄩ€夋渶浼橈紙绛変环浜巆ross_scoring浣嗚緭鍑烘洿绠€娲侊級
            fusion = CrossScoringFusion(obs_3d, obs_t, self.physics)
            result = fusion.fuse()
        else:
            return {'error': f'鏈煡铻嶅悎绛栫暐: {self.fusion_strategy}', 'status': 'FAIL'}

        self.timings['fusion'] = (time.time() - t1) * 1000

        # ============================================
        # Step 2: 鍒ょ綒鍐崇瓥
        # ============================================
        t2 = time.time()
        decision = self._judge(result['landing_x'], result['landing_y'])
        self.timings['decision'] = (time.time() - t2) * 1000

        # ============================================
        # Step 3: 姹囨€�
        # ============================================
        total_time = (time.time() - t_total) * 1000

        output = {
            'status': 'OK',
            'decision': decision,
            'landing_x': result['landing_x'],
            'landing_y': result['landing_y'],
            'landing_t': result.get('landing_t', 0),
            'confidence': result.get('confidence', 0),
            'winning_method': result.get('winning_method', 'unknown'),
            'strategy': self.fusion_strategy,
            'n_trajectory_points': N,
            'observation_duration_s': float(obs_t[-1] - obs_t[0]),
            'timings': self.timings,
            'total_time_ms': total_time,
            'meets_2s_sla': total_time < 2000,
        }

        # 濡傛灉鏈塯round truth锛岃绠楄宸�
        if ground_truth is not None:
            gt_x, gt_y = ground_truth
            error = np.sqrt((result['landing_x'] - gt_x)**2 + 
                           (result['landing_y'] - gt_y)**2)
            output['error_mm'] = float(error * 1000)
            output['correct_decision'] = (
                self._judge(gt_x, gt_y) == decision
            )

        return output

    def _judge(self, x, y):
        """鍒ゆ柇IN/OUT"""
        # 鍗曟墦鍗婂満
        if -5.18 <= x <= 5.18 and -2.59 <= y <= 2.59:
            return 'IN'
        # 鍙屾墦鍗婂満
        if -6.10 <= x <= 6.10 and -3.05 <= y <= 3.05:
            return 'IN (doubles)'
        return 'OUT'


class BatchProcessor:
    """鎵归噺澶勭悊锛堢敤浜庡疄楠岄獙璇侊級"""

    def __init__(self, physics_params=None):
        self.physics = physics_params or {
            'mass': 0.005, 'drag_linear': 0.0, 'drag_quad': 0.001, 'g': 9.81
        }

    def run_all_strategies(self, obs_3d, obs_t, ground_truth=None):
        """瀵逛竴鏉¤建杩硅繍琛屾墍鏈夌瓥鐣�"""
        results = {}

        for strategy in ['cross_scoring', 'precise', 'average']:
            pipeline = HawkEyePipeline(
                physics_params=self.physics,
                fusion_strategy=strategy
            )
            results[strategy] = pipeline.process(obs_3d, obs_t, ground_truth)

        return results

    def run_batch(self, trajectories, ground_truths=None):
        """鎵归噺杩愯"""
        all_results = []
        for i, (obs_3d, obs_t) in enumerate(trajectories):
            gt = ground_truths[i] if ground_truths else None
            results = self.run_all_strategies(obs_3d, obs_t, gt)
            all_results.append(results)
        return all_results


if __name__ == '__main__':
    print("HawkEye Pipeline v3.0 鍔犺浇鎴愬姛銆�")
    print()
    print("鏀寔鐨勮瀺鍚堢瓥鐣�:")
    print("  'cross_scoring' 鈥� 浜ゅ弶璇勫垎铻嶅悎锛堟帹鑽愶紝~150ms锛�")
    print("  'precise'       鈥� 绮剧‘UMTO鍏ㄥ眬浼樺寲锛堢簿搴︽渶楂橈紝~1500ms锛�")
    print("  'average'       鈥� 绠€鍗曞钩鍧囷紙Baseline锛�")
    print("  'best_single'   鈥� 閫夋渶浼樺崟椤癸紙绛変环浜巆ross_scoring锛�")
    print()
    print("浣跨敤鏂瑰紡:")
    print("  pipeline = HawkEyePipeline(fusion_strategy='cross_scoring')")
    print("  result = pipeline.process(obs_3d, obs_t)")