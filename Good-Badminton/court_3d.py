"""
3D 羽毛球场可视化 (matplotlib 版) — Hawk-Eye 风格
真实 RK4 物理轨迹 | 鹰眼俯视动画 | 落点压痕
"""

import numpy as np
import matplotlib
import os as _os
# 默认无窗口后端：避免在脚本/服务里 plt.show() 阻塞整个流程
# 需要弹出交互窗口时设环境变量 HAWKEYE_GUI=1
if _os.environ.get("HAWKEYE_GUI") == "1":
    try:
        matplotlib.use("TkAgg")
    except Exception:
        matplotlib.use("Agg")
else:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.widgets import Button
import threading
import time
import math

# ======== 中文字体 ========
_USE_FONT = None
for _f in ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'Noto Sans SC']:
    try:
        _p = fm.findfont(_f, fallback_to_default=False)
        if _p:
            _USE_FONT = _f
            break
    except:
        pass

if _USE_FONT:
    plt.rcParams['font.sans-serif'] = [_USE_FONT, 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    _LANG = 'cn'
else:
    _LANG = 'en'

# ======== 场地常量 ========
L, W = 13.40, 6.10
W_SINGLES = 5.18
HL, HW = L / 2, W / 2
HW_S = W_SINGLES / 2
SERVICE_FRONT = 1.98
SERVICE_BACK = 0.72
NET_H = 1.55
POST_SPREAD = 0.5
POST_Y = HW + POST_SPREAD

# ======== Z-ORDER ========
Z_FLOOR = 1
Z_NET = 6
Z_TRAJ_BEHIND = 4
Z_TRAJ_FRONT = 50
Z_MARK = 97
Z_LINES = 100

COLORS = {
    "china": {
        "floor": "#2d8a4e", "outer": "#1a4a28",
        "line": "#ffffff",
        "net": "#555555", "net_top": "#eeeeee",
        "post": "#aaaaaa",
        "bg": "#1a1a2e",
        "axis": "#8899aa",
        "panel_bg": "#0d1117e8",
        "panel_border": "#334455",
    },
}

_TXT = {
    'cn': {
        'lx': 'X 长边 (m)', 'ly': 'Y 宽边 (m)', 'lz': 'Z 高度 (m)',
        'title': '羽毛球鹰眼系统',
        'hint': '左键旋转 | 右键平移 | 滚轮缩放 | R 恢复全景',
        'btn_return': '恢复全景',
        'btn_replay': '回放',
    },
    'en': {
        'lx': 'X (m)', 'ly': 'Y (m)', 'lz': 'Z (m)',
        'title': 'Hawk-Eye',
        'hint': 'L-drag: rotate | R-drag: pan | Scroll: zoom | R: reset',
        'btn_return': 'Reset',
        'btn_replay': 'Replay',
    },
}
_T = _TXT.get(_LANG, _TXT['en'])


class Court3D:

    def __init__(self, scheme="china", figsize=(15, 9)):
        c = COLORS.get(scheme, COLORS["china"])
        self._c = c
        self._default_view = {'elev': 28, 'azim': -55, 'dist': None}
        self._net_zorder = Z_NET
        self._traj_lines = []
        self._mark_elements = []
        self._info_elements = []
        self._landing_xy = None
        self._traj_pts = None
        self._zoomed = False

        self.fig = plt.figure(figsize=figsize, facecolor=c["bg"])
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_facecolor(c["bg"])

        self._build()
        self._setup_view()
        self._default_view['dist'] = self._get_dist()

        self._base_box = self.ax.get_box_aspect()
        self._base_xl = self.ax.get_xlim()
        self._base_yl = self.ax.get_ylim()
        self._base_zl = self.ax.get_zlim()

        self._hint = self.fig.text(
            0.5, 0.012, _T['hint'], color='#556677', fontsize=8,
            ha='center', va='center', zorder=300)

        # 按钮
        self._btn_ax1 = self.fig.add_axes([0.33, 0.005, 0.12, 0.04])
        self._btn1 = Button(self._btn_ax1, _T['btn_return'],
                            color='#2a3a4a', hovercolor='#3a5a6a')
        self._btn1.label.set_color('#aabbcc')
        self._btn1.label.set_fontsize(9)
        if _USE_FONT:
            self._btn1.label.set_fontproperties(fm.FontProperties(family=_USE_FONT, size=9))
        self._btn1.on_clicked(lambda e: self.reset_view())

        self._btn_ax2 = self.fig.add_axes([0.47, 0.005, 0.12, 0.04])
        self._btn2 = Button(self._btn_ax2, _T['btn_replay'],
                            color='#2a3a4a', hovercolor='#3a5a6a')
        self._btn2.label.set_color('#aabbcc')
        self._btn2.label.set_fontsize(9)
        if _USE_FONT:
            self._btn2.label.set_fontproperties(fm.FontProperties(family=_USE_FONT, size=9))
        self._btn2.on_clicked(lambda e: self._replay())

# self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.fig.canvas.mpl_connect('key_press_event', lambda e: self.reset_view() if e.key == 'r' else None)

    # ── 相机 ──
    def _get_dist(self):
        return float(getattr(self.ax, 'dist', getattr(self.ax, '_dist', 10.0)))

    def _set_dist(self, v):
        try: self.ax.dist = v
        except AttributeError: self.ax._dist = v


    # ── 绘图 ──
    def _rect(self, x1, y1, x2, y2, z, c, a=1.0):
        self.ax.add_collection3d(Poly3DCollection(
            [[(x1, y1, z), (x2, y1, z), (x2, y2, z), (x1, y2, z)]],
            alpha=a, facecolor=c, edgecolor='none', zorder=Z_FLOOR))

    def _cl(self, xs, ys, z, c, lw):
        self.ax.plot(xs, ys, [z, z], color=c, linewidth=lw,
                     solid_capstyle='round', zorder=Z_LINES, alpha=0.95)

    def _nl(self, xs, ys, zs, c, lw=1, a=0.6):
        self.ax.plot(xs, ys, zs, color=c, linewidth=lw, alpha=a,
                     zorder=self._net_zorder)

    # ── 场地 ──
    def _build(self):
        c = self._c; w = c["line"]; LZ = 0.018
        self._rect(-HL, -HW, HL, HW, 0, c["floor"])
        self._rect(-HL, HW, HL, POST_Y, 0, c["outer"])
        self._rect(-HL, -POST_Y, HL, -HW, 0, c["outer"])
        for y in [-HW, HW]:     self._cl([-HL, HL], [y, y], LZ, w, 3.5)
        for y in [-HW_S, HW_S]: self._cl([-HL, HL], [y, y], LZ, w, 2.2)
        for x in [-HL, HL]:     self._cl([x, x], [-HW, HW], LZ, w, 3.5)
        for x in [-SERVICE_FRONT, SERVICE_FRONT]:
            self._cl([x, x], [-HW, HW], LZ, w, 2.2)
        for x in [-(HL - SERVICE_BACK), (HL - SERVICE_BACK)]:
            self._cl([x, x], [-HW, HW], LZ, w, 1.8)
        for s in [-1, 1]: self._cl([s * HL, s * SERVICE_FRONT], [0, 0], LZ, w, 1.8)

        nc = c["net"]
        self.ax.add_collection3d(Poly3DCollection(
            [[(0, -POST_Y, 0), (0, POST_Y, 0), (0, POST_Y, NET_H), (0, -POST_Y, NET_H)]],
            alpha=0.38, facecolor=nc, edgecolor=nc, zorder=self._net_zorder))
        for yy in np.linspace(-POST_Y, POST_Y, 25):
            self._nl([0, 0], [yy, yy], [0.01, NET_H * 0.95], nc, 0.5, 0.4)
        for hh in np.linspace(0.2, NET_H * 0.9, 8):
            self._nl([0, 0], [-POST_Y, POST_Y], [hh, hh], nc, 0.4, 0.4)
        self._cl([0, 0], [-POST_Y, POST_Y], NET_H, c["net_top"], 4.5)
        for s in [-1, 1]:
            py = s * POST_Y
            self.ax.plot([0, 0], [py, py], [0, NET_H + 0.15], color=c["post"],
                         lw=6, solid_capstyle='round', zorder=12)
            self.ax.scatter([0], [py], [NET_H + 0.15], c=c["net_top"], s=45, zorder=15, edgecolors='none')

        ac = c["axis"]
        self.ax.set_xlabel(_T['lx'], color=ac, fontsize=9, labelpad=8)
        self.ax.set_ylabel(_T['ly'], color=ac, fontsize=9, labelpad=8)
        self.ax.set_zlabel(_T['lz'], color=ac, fontsize=9, labelpad=8)
        self.ax.set_title(_T['title'], color='#e0e0e0', fontsize=14, fontweight='bold', pad=10)
        for p in [self.ax.xaxis.pane, self.ax.yaxis.pane, self.ax.zaxis.pane]:
            p.fill = True; p.set_facecolor('#1c2530'); p.set_edgecolor('#1e2a34'); p.set_alpha(0.6)
        self.ax.grid(True, alpha=0.06, color='#1e2a34')
        self.ax.tick_params(colors=ac, labelsize=8)

    def _setup_view(self):
        self.ax.set_xlim(-HL - 1.5, HL + 1.5)
        self.ax.set_ylim(-POST_Y - 0.3, POST_Y + 0.3)
        self.ax.set_zlim(0, 6)
        self.ax.view_init(elev=28, azim=-55)
        self.ax.set_box_aspect([L + 3, W + 3, 6])

    # ── 清除 ──
    def _rm_traj(self):
        for l in self._traj_lines:
            try: l.remove()
            except: pass
        self._traj_lines.clear()

    def _rm_mark(self):
        for e in self._mark_elements:
            try: e.remove()
            except: pass
        self._mark_elements.clear()

    def _rm_info(self):
        for e in self._info_elements:
            try: e.remove()
            except: pass
        self._info_elements.clear()

    # ── 轨迹 ──
    def update_trajectory(self, pts, color='#00e676', lw=2.5):
        self._rm_traj()
        self._traj_pts = pts.copy()
        if len(pts) < 2: return
        m = pts[:, 2] >= 0; p = pts[m]
        if len(p) < 2: return
        left = p[p[:, 0] <= 0.05]; right = p[p[:, 0] >= -0.05]
        if len(left) >= 2:
            h, = self.ax.plot(left[:, 0], left[:, 1], left[:, 2],
                              color=color, lw=lw, zorder=Z_TRAJ_BEHIND, alpha=0.4)
            self._traj_lines.append(h)
        if len(right) >= 2:
            h, = self.ax.plot(right[:, 0], right[:, 1], right[:, 2],
                              color=color, lw=lw, zorder=Z_TRAJ_FRONT, alpha=1.0)
            self._traj_lines.append(h)
        last = p[-1]
        sc = self.ax.scatter(*last, c=color, s=40, edgecolors='white', linewidth=0.3, zorder=55)
        self._traj_lines.append(sc)

    # ── 鹰眼落点压痕（小椭圆）──
    def mark_landing(self, x, y):
        """
        鹰眼风格球头压痕 — 小椭圆白色印记。
        约 4cm × 3cm，模拟真实球头触地变形。
        使用 plot 小圆点，不做光环。
        """
        self._rm_mark()
        self._landing_xy = (x, y)
        # 主体：小白点
        d, = self.ax.plot([x], [y], [0.028], marker='o',
                          color='#fafafa', markersize=8,
                          markeredgewidth=0, zorder=Z_MARK, alpha=0.95)
        self._mark_elements.append(d)

    # ── 右上角信息 ──
    def show_info(self, data=None):
        self._rm_info()
        if data is None: data = {}

        y0 = 0.94
        if 'ruling' in data:
            ruling = data['ruling']
            is_out = 'OUT' in ruling.upper()
            rc = '#ff5555' if is_out else '#55ff88'
            kw = dict(color=rc, ha='right', va='top', zorder=400,
                      bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS["china"]["panel_bg"],
                                edgecolor=COLORS["china"]["panel_border"], linewidth=0.8))
            if _USE_FONT:
                kw['fontproperties'] = fm.FontProperties(family=_USE_FONT, size=16, weight='bold')
            else:
                kw['fontsize'] = 16; kw['fontweight'] = 'bold'
            t = self.fig.text(0.96, y0, ruling, **kw)
            self._info_elements.append(t)
            y0 -= 0.055

        entries = []
        if 'speed' in data: entries.append(f"球速  {data['speed']}")
        if 'entry_angle' in data: entries.append(f"入角  {data['entry_angle']}")
        if 'flight_time' in data: entries.append(f"飞行  {data['flight_time']}")
        if 'distance' in data: entries.append(f"距离  {data['distance']}")
        if 'max_height' in data: entries.append(f"最高  {data['max_height']}")

        if entries:
            text = '\n'.join(entries)
            kw2 = dict(color='#aabbcc', ha='right', va='top', zorder=400)
            if _USE_FONT:
                kw2['fontproperties'] = fm.FontProperties(family=_USE_FONT, size=9)
            else:
                kw2['fontsize'] = 9
            t2 = self.fig.text(0.96, y0, text, **kw2)
            self._info_elements.append(t2)

    # ── 鹰眼下降 ──
    def zoom_to_landing(self, x, y, radius=0.7, ms=1200):
        self._zoomed = True
        ry = radius * (self._base_yl[1] - self._base_yl[0]) / (self._base_xl[1] - self._base_xl[0])
        tx = (x - radius, x + radius); ty = (y - ry, y + ry); tz = (0, 0.4)
        sx, sy, sz = self.ax.get_xlim(), self.ax.get_ylim(), self.ax.get_zlim()
        sd, se, sa = self._get_dist(), float(self.ax.elev), float(self.ax.azim)
        p1 = max(8, ms // 4 // 33); p2 = max(15, ms * 3 // 4 // 33)
        total = p1 + p2

        def _anim(f):
            if f < p1:
                t = (f + 1) / p1; t = t * t * (3 - 2 * t)
                self.ax.elev = se + (55 - se) * t
                self.ax.azim = sa + (-90 - sa) * t * 0.3
                m = t * 0.3
                self.ax.set_xlim(sx[0] + (tx[0] - sx[0]) * m, sx[1] + (tx[1] - sx[1]) * m)
                self.ax.set_ylim(sy[0] + (ty[0] - sy[0]) * m, sy[1] + (ty[1] - sy[1]) * m)
                self.ax.set_zlim(sz[0] + (tz[0] - sz[0]) * m, sz[1] + (tz[1] - sz[1]) * m)
                self._set_dist(sd + (radius * 2.0 - sd) * m)
            else:
                t = (f - p1 + 1) / p2; t = t * t * (3 - 2 * t)
                self.ax.elev = 55 + (90 - 55) * t; self.ax.azim = -90
                self.ax.set_xlim(sx[0] + (tx[0] - sx[0]) * t, sx[1] + (tx[1] - sx[1]) * t)
                self.ax.set_ylim(sy[0] + (ty[0] - sy[0]) * t, sy[1] + (ty[1] - sy[1]) * t)
                self.ax.set_zlim(sz[0] + (tz[0] - sz[0]) * t, sz[1] + (tz[1] - sz[1]) * t)
                self._set_dist(sd + (radius * 1.5 - sd) * t)
            self.ax.set_box_aspect(self._base_box)
            # 每帧重绘白点
            self._rm_mark()
            d, = self.ax.plot([x], [y], [0.028], marker='o',
                              color='#fafafa', markersize=8,
                              markeredgewidth=0, zorder=Z_MARK, alpha=0.95)
            self._mark_elements.append(d)
            return []

        def _done(*_a):
            self._rm_traj()
            self._rm_mark()
            d, = self.ax.plot([x], [y], [0.028], marker='o',
                              color='#fafafa', markersize=10,
                              markeredgewidth=0, zorder=Z_MARK, alpha=1.0)
            self._mark_elements.append(d)
            self.fig.canvas.draw_idle()

        self._anim = animation.FuncAnimation(self.fig, _anim, frames=total, interval=33, blit=False, repeat=False)
        tmr = self.fig.canvas.new_timer(interval=total * 33 + 120)
        tmr.single_shot = True; tmr.add_callback(_done); tmr.start()

    def reset_view(self, ms=500):
        self._zoomed = False
        tx, ty, tz = self._base_xl, self._base_yl, self._base_zl
        td, te, ta = self._default_view['dist'], self._default_view['elev'], self._default_view['azim']
        sx, sy, sz = self.ax.get_xlim(), self.ax.get_ylim(), self.ax.get_zlim()
        sd, se, sa = self._get_dist(), float(self.ax.elev), float(self.ax.azim)
        frames = max(8, ms // 40)

        def _anim(f):
            t = (f + 1) / frames; t = t * t * (3 - 2 * t)
            self.ax.set_xlim(sx[0] + (tx[0] - sx[0]) * t, sx[1] + (tx[1] - sx[1]) * t)
            self.ax.set_ylim(sy[0] + (ty[0] - sy[0]) * t, sy[1] + (ty[1] - sy[1]) * t)
            self.ax.set_zlim(sz[0] + (tz[0] - sz[0]) * t, sz[1] + (tz[1] - sz[1]) * t)
            self._set_dist(sd + (td - sd) * t)
            self.ax.elev = se + (te - se) * t; self.ax.azim = sa + (ta - sa) * t
            self.ax.set_box_aspect(self._base_box)
            return []

        def _done(*_a):
            if self._traj_pts is not None:
                self._rm_traj()
                self.update_trajectory(self._traj_pts)
                if self._landing_xy:
                    self._rm_mark()
                    x, y = self._landing_xy
                    d, = self.ax.plot([x], [y], [0.028], marker='o',
                                      color='#fafafa', markersize=8,
                                      markeredgewidth=0, zorder=Z_MARK, alpha=0.95)
                    self._mark_elements.append(d)
            self.fig.canvas.draw_idle()

        self._anim = animation.FuncAnimation(self.fig, _anim, frames=frames, interval=ms // frames, blit=False, repeat=False)
        tmr = self.fig.canvas.new_timer(interval=frames * ms // frames + 80)
        tmr.single_shot = True; tmr.add_callback(_done); tmr.start()

    def _replay(self):
        self.reset_view(300)
        def _r():
            time.sleep(0.4)
            if self._traj_pts is not None:
                self.update_trajectory(self._traj_pts)
            if self._landing_xy:
                self.mark_landing(*self._landing_xy)
            self.fig.canvas.draw_idle()
        threading.Thread(target=_r, daemon=True).start()

    def save(self, path="court_3d.png", dpi=200):
        self.fig.savefig(path, dpi=dpi, facecolor=self._c["bg"], bbox_inches='tight')
        plt.close(self.fig)
        print(f"[OK] {path}")

    def show(self, save_if_headless: str = "outputs/court_3d.png"):
        plt.tight_layout()
        if matplotlib.get_backend().lower() == "agg":
            # 无窗口后端：保存成图片，绝不阻塞
            _os.makedirs(_os.path.dirname(save_if_headless) or ".", exist_ok=True)
            self.fig.savefig(save_if_headless, dpi=160,
                             facecolor=self._c["bg"], bbox_inches="tight")
            plt.close(self.fig)
            print(f"[OK] 无窗口环境，3D 视图已保存: {save_if_headless}")
            return
        plt.show()


# ============================================================
if __name__ == '__main__':
    from shuttlecock_physics import ShuttlecockPhysics

    print(f"Hawk-Eye | 字体: {_USE_FONT or '无中文字体'}")

    court = Court3D(scheme="china")

    # ── 刚好出双打边线 3-5cm ──
    # 双打边线 y=3.05m，目标 y≈3.08~3.10
    phys = ShuttlecockPhysics()
    ang = math.radians(-3.0)
    pos0 = np.array([-5.5, -0.5, 2.8])
    # 横向 18m/s 确保飞出去
    vel0 = np.array([65.0 * math.cos(ang), 18.0, 65.0 * math.sin(ang)])

    landing = phys.predict_landing(pos0, vel0, z_ground=0.0,
                                   return_trajectory=True)
    traj = phys._cache_trajectory
    pts = np.array([[p.x, p.y, p.z] for p in traj if p.z >= 0])

    # 判罚
    is_in_x = abs(landing.x) <= HL
    is_in_y = abs(landing.y) <= HW
    dy_out = abs(landing.y) - HW
    if is_in_x and is_in_y:
        ruling = 'IN'
    elif not is_in_x:
        ruling = f'OUT (底线+{abs(landing.x)-HL:.2f}m)'
    else:
        ruling = f'OUT (边线+{dy_out*100:.1f}cm)'

    ci = np.argmin(np.abs(pts[:, 0]))
    print(f"RK4: 过网 z={pts[ci,2]:.2f}m | "
          f"落地({landing.x:.3f},{landing.y:.3f})m | "
          f"边线+{dy_out*100:.1f}cm | {landing.speed*3.6:.0f}km/h | {ruling}")

    court.update_trajectory(pts, '#00e676')
    court.mark_landing(landing.x, landing.y)
    court.show_info({
        'ruling': ruling,
        'speed': f'{landing.speed * 3.6:.0f} km/h',
        'entry_angle': f'{math.degrees(math.atan2(-landing.vz, math.sqrt(landing.vx**2+landing.vy**2))):.1f}°',
        'flight_time': f'{landing.t:.2f} s',
        'distance': f'{math.sqrt(landing.x**2 + landing.y**2):.2f} m',
        'max_height': f'{max(p.z for p in traj):.2f} m',
    })

    def _demo():
        time.sleep(2.5)
        print("鹰眼下降...")
        court.zoom_to_landing(landing.x, landing.y, radius=0.5)

    threading.Thread(target=_demo, daemon=True).start()
    court.show()