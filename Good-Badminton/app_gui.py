"""
app_gui.py — 羽毛球鹰眼分析台（桌面窗口版）
=============================================
用法：
    python app_gui.py          # 或双击 启动.bat

功能：
    · 把视频拖进窗口 → 自动识别球场 / 追踪羽毛球 / 估算落点 / 判界内界外
    · 一键跑"模拟全链路"（不需要相机和视频，物理引擎合成轨迹 + 双目重建）
    · 分析结果：窗口内预览 + 标注视频 + result.json

依赖：numpy、opencv-python、pillow、windnd（拖拽），全部已装在项目 .venv 里
"""

import os
import queue
import subprocess
import sys
import threading
import traceback

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

try:
    from PIL import Image, ImageTk
except Exception:
    Image = ImageTk = None

try:
    import windnd
except Exception:
    windnd = None

VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".flv", ".wmv")

C_BG = "#12161c"
C_PANEL = "#1b2129"
C_CARD = "#232b35"
C_TEXT = "#e8edf2"
C_SUB = "#8b98a6"
C_ACCENT = "#4a9eff"
C_OK = "#5fd18b"
C_WARN = "#f0b357"
C_ERR = "#ef6b6b"


class HawkEyeApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.video_path = None
        self.out_dir = None
        self.annotated = None
        self.q = queue.Queue()
        self.busy = False
        self._preview_img = None
        self.var_mode = tk.StringVar(value="双打")
        self.var_frames = tk.StringVar(value="600")
        self.var_detector = tk.StringVar(value="自动")
        self.manual_corners = None

        root.title("羽毛球鹰眼分析台 · Badminton Hawk-Eye")
        root.geometry("1060x800")
        root.minsize(900, 700)
        root.configure(bg=C_BG)

        self._build_ui()
        self._hook_drop()
        self.root.after(120, self._poll_queue)

    # ══════════════════════════════════════════
    # UI
    # ══════════════════════════════════════════
    def _build_ui(self):
        # 顶部标题
        head = tk.Frame(self.root, bg=C_BG)
        head.pack(fill="x", padx=22, pady=(18, 6))
        tk.Label(head, text="羽毛球鹰眼分析台", bg=C_BG, fg=C_TEXT,
                 font=("Microsoft YaHei UI", 20, "bold")).pack(side="left")
        tk.Label(head, text="拖入视频 → 自动识别球场 · 追踪羽球 · 判界内外",
                 bg=C_BG, fg=C_SUB, font=("Microsoft YaHei UI", 10)).pack(side="left", padx=(14, 0), pady=(8, 0))

        # 拖拽区
        self.drop = tk.Canvas(self.root, bg=C_PANEL, highlightthickness=0, height=132)
        self.drop.pack(fill="x", padx=22, pady=10)
        self.drop.bind("<Configure>", lambda e: self._draw_dropzone())
        self.drop.bind("<Button-1>", lambda e: self.select_file())
        self._draw_dropzone()

        # 按钮行
        bar = tk.Frame(self.root, bg=C_BG)
        bar.pack(fill="x", padx=22)
        self.btn_choose = self._btn(bar, "选择视频", self.select_file, kind="ghost")
        self.btn_run = self._btn(bar, "开始分析", self.start_analysis, kind="primary")
        self.btn_sim = self._btn(bar, "模拟全链路演示", self.run_simulate, kind="ghost")
        self.btn_open = self._btn(bar, "打开输出目录", self.open_out, kind="ghost")
        for b in (self.btn_choose, self.btn_run, self.btn_sim, self.btn_open):
            b.pack(side="left", padx=(0, 10))

        self.status = tk.Label(bar, text="就绪", bg=C_BG, fg=C_SUB,
                               font=("Microsoft YaHei UI", 10))
        self.status.pack(side="right")

        # 选项行
        opt = tk.Frame(self.root, bg=C_BG)
        opt.pack(fill="x", padx=22, pady=(8, 0))
        tk.Label(opt, text="判罚模式", bg=C_BG, fg=C_SUB,
                 font=("Microsoft YaHei UI", 9)).pack(side="left")
        cb = ttk.Combobox(opt, textvariable=self.var_mode, values=["双打", "单打"],
                          width=6, state="readonly")
        cb.pack(side="left", padx=(6, 18))
        tk.Label(opt, text="最多处理帧数", bg=C_BG, fg=C_SUB,
                 font=("Microsoft YaHei UI", 9)).pack(side="left")
        tk.Entry(opt, textvariable=self.var_frames, width=8, bg=C_CARD, fg=C_TEXT,
                 insertbackground=C_TEXT, relief="flat").pack(side="left", padx=(6, 18))
        tk.Label(opt, text="检测器", bg=C_BG, fg=C_SUB,
                 font=("Microsoft YaHei UI", 9)).pack(side="left")
        ttk.Combobox(opt, textvariable=self.var_detector, values=["自动", "运动检测", "YOLO"],
                     width=8, state="readonly").pack(side="left", padx=(6, 18))
        self.btn_corner = self._btn(opt, "手动标定球场", self.open_corner_picker, kind="ghost")
        self.btn_corner.pack(side="left", padx=(0, 10))
        self.btn_clear = self._btn(opt, "清除标定", self.clear_calib, kind="ghost")
        self.btn_clear.pack(side="left", padx=(0, 10))
        self.btn_play = self._btn(opt, "播放标注视频", self.open_video, kind="ghost")
        self.btn_play.pack(side="left")

        # 主体：左日志 右预览
        body = tk.Frame(self.root, bg=C_BG)
        body.pack(fill="both", expand=True, padx=22, pady=(12, 6))

        left = tk.Frame(body, bg=C_PANEL)
        left.pack(side="left", fill="both", expand=True)
        tk.Label(left, text="运行日志", bg=C_PANEL, fg=C_SUB,
                 font=("Microsoft YaHei UI", 9)).pack(anchor="w", padx=12, pady=(8, 0))
        wrap = tk.Frame(left, bg=C_PANEL)
        wrap.pack(fill="both", expand=True, padx=8, pady=8)
        self.log_box = tk.Text(wrap, bg="#0e1318", fg=C_TEXT, insertbackground=C_TEXT,
                               relief="flat", font=("Consolas", 10), wrap="word",
                               padx=10, pady=8, height=18)
        sb = tk.Scrollbar(wrap, command=self.log_box.yview, bg=C_PANEL)
        self.log_box.configure(yscrollcommand=sb.set)
        self.log_box.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        for tag, color in (("ok", C_OK), ("warn", C_WARN), ("err", C_ERR),
                           ("info", C_TEXT), ("dim", C_SUB)):
            self.log_box.tag_config(tag, foreground=color)

        right = tk.Frame(body, bg=C_PANEL, width=460)
        right.pack(side="right", fill="both", padx=(12, 0))
        tk.Label(right, text="分析预览", bg=C_PANEL, fg=C_SUB,
                 font=("Microsoft YaHei UI", 9)).pack(anchor="w", padx=12, pady=(8, 0))
        self.preview = tk.Label(right, bg="#0e1318", fg=C_SUB,
                                text="分析完成后在此显示预览图", font=("Microsoft YaHei UI", 10))
        self.preview.pack(fill="both", expand=True, padx=8, pady=8)

        self._log("欢迎使用。把视频拖进上方区域，或点「选择视频」。", "dim")
        self._log("没有现成视频？点「模拟全链路演示」即可用物理引擎合成轨迹跑一遍。", "dim")

    def _btn(self, parent, text, cmd, kind="ghost"):
        if kind == "primary":
            bg, fg, active = C_ACCENT, "#0b1116", "#63b0ff"
        else:
            bg, fg, active = C_CARD, C_TEXT, "#2d3742"
        b = tk.Button(parent, text=text, command=cmd, bg=bg, fg=fg,
                      activebackground=active, activeforeground=fg,
                      relief="flat", bd=0, padx=16, pady=8, cursor="hand2",
                      font=("Microsoft YaHei UI", 10, "bold" if kind == "primary" else "normal"))
        b.bind("<Enter>", lambda e: b.config(bg=active) if str(b["state"]) != "disabled" else None)
        b.bind("<Leave>", lambda e: b.config(bg=bg) if str(b["state"]) != "disabled" else None)
        return b

    def _draw_dropzone(self):
        c = self.drop
        c.delete("all")
        w = c.winfo_width() or 1000
        h = c.winfo_height() or 132
        c.create_rectangle(3, 3, w - 3, h - 3, outline="#33404f", width=2, dash=(7, 5))
        name = os.path.basename(self.video_path) if self.video_path else None
        if name:
            c.create_text(w / 2, h / 2 - 14, text=f"已选择：{name}", fill=C_OK,
                          font=("Microsoft YaHei UI", 13, "bold"))
            c.create_text(w / 2, h / 2 + 14, text="点此更换视频，或直接拖入新文件",
                          fill=C_SUB, font=("Microsoft YaHei UI", 9))
        else:
            c.create_text(w / 2, h / 2 - 16, text="把羽毛球视频拖到这里",
                          fill=C_TEXT, font=("Microsoft YaHei UI", 15, "bold"))
            c.create_text(w / 2, h / 2 + 16, text="支持 mp4 / mov / avi / mkv   ·   也可以点击此处选择文件",
                          fill=C_SUB, font=("Microsoft YaHei UI", 9))

    def _hook_drop(self):
        if windnd is None:
            self._log("未安装 windnd，拖拽不可用（仍可用「选择视频」按钮）", "warn")
            return
        try:
            hwnd = self.root.winfo_id()
            try:  # Windows 下需挂到顶层窗口句柄上，覆盖整窗拖放
                import ctypes
                parent = ctypes.windll.user32.GetParent(hwnd)
                if parent:
                    hwnd = parent
            except Exception:
                pass
            windnd.hook_dropfiles(hwnd, func=self._on_drop)
            self._log("拖拽已启用：把视频文件拖进窗口即可。", "ok")
        except Exception as e:
            self._log(f"拖拽注册失败（仍可用「选择视频」）：{e}", "warn")

    def _on_drop(self, files):
        if not files:
            return
        for raw in files:
            path = raw.decode("gbk", errors="ignore") if isinstance(raw, bytes) else str(raw)
            path = path.strip().strip('"')
            if os.path.isdir(path):
                for f in os.listdir(path):
                    if f.lower().endswith(VIDEO_EXT):
                        path = os.path.join(path, f)
                        break
            if path.lower().endswith(VIDEO_EXT):
                self.root.after(0, lambda p=path: self._set_video(p))
                return
        self.root.after(0, lambda: self._log("拖入的不是视频文件（需 mp4/mov/avi/mkv）", "err"))

    # ══════════════════════════════════════════
    # 交互
    # ══════════════════════════════════════════
    def _set_video(self, path):
        self.video_path = path
        self._draw_dropzone()
        self._log(f"已选择视频：{path}", "ok")
        self.status.config(text="已就绪，点「开始分析」", fg=C_ACCENT)

    def select_file(self):
        p = filedialog.askopenfilename(
            title="选择羽毛球视频",
            filetypes=[("视频文件", "*.mp4 *.mov *.avi *.mkv *.m4v *.flv"), ("所有文件", "*.*")])
        if p:
            self._set_video(p)

    def start_analysis(self):
        if self.busy:
            return
        if not self.video_path:
            messagebox.showinfo("提示", "先拖一个视频进来，或点「选择视频」。")
            return
        self.busy = True
        self._toggle_buttons(False)
        self.status.config(text="分析中…", fg=C_WARN)
        threading.Thread(target=self._worker_analyze, daemon=True).start()

    def _worker_analyze(self):
        try:
            from video_pipeline import analyze_video
            mode = "singles" if self.var_mode.get() == "单打" else "doubles"
            try:
                max_frames = int(self.var_frames.get() or 600)
            except ValueError:
                max_frames = 600
            det_map = {"自动": "auto", "运动检测": "motion", "YOLO": "yolo"}
            res = analyze_video(self.video_path, mode=mode, max_frames=max_frames,
                                detector=det_map.get(self.var_detector.get(), "auto"),
                                manual_corners=self.manual_corners,
                                log=lambda m: self.q.put(("log", m)))
            if res.get("preview"):
                self.q.put(("preview", res["preview"]))
            self.out_dir = res.get("out_dir")
            self.annotated = res.get("annotated_video")
            self.q.put(("done", res))
        except Exception:
            self.q.put(("log", "✗ 分析异常：\n" + traceback.format_exc()))
            self.q.put(("done", {"ok": False}))

    def run_simulate(self):
        if self.busy:
            return
        self.busy = True
        self._toggle_buttons(False)
        self.status.config(text="模拟运行中…", fg=C_WARN)
        threading.Thread(target=self._worker_simulate, daemon=True).start()

    def _worker_simulate(self):
        try:
            self.q.put(("log", "▶ 模拟全链路：物理引擎合成轨迹 + 13.4m 大基线双目重建 + 落地判罚"))
            proc = subprocess.Popen(
                [sys.executable, "-u", os.path.join(BASE, "run_full_demo.py"), "--mode", "simulate"],
                cwd=BASE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="ignore")
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    self.q.put(("log", "  " + line))
            proc.wait()
            self.q.put(("log", "✔ 模拟完成"))
            self.q.put(("done", {"ok": True, "out_dir": BASE}))
        except Exception:
            self.q.put(("log", "✗ 模拟失败：\n" + traceback.format_exc()))
            self.q.put(("done", {"ok": False}))

    def open_corner_picker(self):
        """弹出首帧，让用户依次点击球场四角（左上/右上/右下/左下）。"""
        if not self.video_path:
            messagebox.showinfo("提示", "先选一个视频。")
            return
        try:
            import cv2
        except Exception:
            messagebox.showerror("错误", "缺少 opencv，无法标定。")
            return
        cap = cv2.VideoCapture(self.video_path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(int(n * 0.1), 0))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            cap = cv2.VideoCapture(self.video_path)
            ok, frame = cap.read()
            cap.release()
        if not ok or Image is None:
            messagebox.showerror("错误", "读取首帧失败。")
            return

        h, w = frame.shape[:2]
        scale = min(900.0 / w, 620.0 / h, 1.0)
        disp = cv2.resize(frame, None, fx=scale, fy=scale)
        win = tk.Toplevel(self.root)
        win.title("点击球场四角：左上 → 右上 → 右下 → 左下")
        win.configure(bg=C_BG)
        cv = tk.Canvas(win, width=disp.shape[1], height=disp.shape[0], highlightthickness=0)
        cv.pack()
        img = Image.fromarray(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))
        photo = ImageTk.PhotoImage(img)
        cv.create_image(0, 0, anchor="nw", image=photo)
        cv.image = photo
        tk.Label(win, text="依次点击球场四个角，点错可按 Backspace 撤销；点满四点自动保存",
                 bg=C_BG, fg=C_SUB, font=("Microsoft YaHei UI", 9)).pack(fill="x", pady=6)
        pts = []

        def redraw():
            cv.delete("mk")
            for i, (x, y) in enumerate(pts):
                cv.create_oval(x - 8, y - 8, x + 8, y + 8, outline=C_ACCENT, width=2, tags="mk")
                cv.create_text(x + 14, y, text=str(i + 1), fill=C_ACCENT,
                               font=("Microsoft YaHei UI", 11, "bold"), tags="mk")

        def on_click(e):
            if len(pts) >= 4:
                return
            pts.append((e.x, e.y))
            redraw()
            if len(pts) == 4:
                self.manual_corners = [[x / scale, y / scale] for x, y in pts]
                self._log(f"手动标定完成：{[f'({int(x)},{int(y)})' for x, y in self.manual_corners]}", "ok")
                self._log("重新点「开始分析」即用手动四点做落点判罚。", "dim")
                win.destroy()

        def on_undo(_e):
            if pts:
                pts.pop()
                redraw()

        cv.bind("<Button-1>", on_click)
        win.bind("<BackSpace>", on_undo)
        win.transient(self.root)
        win.grab_set()

    def clear_calib(self):
        """删除已保存的单机位标定（换机位/换场地时先清一次）。"""
        import os as _os
        try:
            from video_pipeline import DEFAULT_CALIB
        except Exception:
            DEFAULT_CALIB = "calib_single_cam.json"
        self.manual_corners = None
        if _os.path.exists(DEFAULT_CALIB):
            try:
                _os.remove(DEFAULT_CALIB)
                self._log(f"已删除保存的标定：{DEFAULT_CALIB}", "ok")
            except Exception as e:
                self._log(f"删除失败：{e}", "err")
        else:
            self._log("当前没有保存的标定文件。", "dim")
        self._log("下次「开始分析」将重新自动检测球场，或用「手动标定球场」点四个角。", "dim")

    def open_out(self):
        target = self.out_dir or BASE
        try:
            os.startfile(target)          # Windows
        except Exception:
            subprocess.Popen(["explorer", target])

    def open_video(self):
        if self.annotated and os.path.exists(self.annotated):
            try:
                os.startfile(self.annotated)
                return
            except Exception:
                pass
        messagebox.showinfo("提示", "还没有标注视频，先跑一次「开始分析」。")

    def _toggle_buttons(self, enabled):
        state = "normal" if enabled else "disabled"
        for b in (self.btn_choose, self.btn_run, self.btn_sim):
            b.config(state=state)

    # ══════════════════════════════════════════
    # 消息泵
    # ══════════════════════════════════════════
    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "log":
                    self._log(payload)
                    if "进度" in payload and not payload.strip().startswith("  [检测] 处理"):
                        try:
                            pct = payload.split("进度")[1].split("%")[0].strip()
                            self.status.config(text=f"分析中… {pct}%", fg=C_WARN)
                        except Exception:
                            pass
                elif kind == "preview":
                    self._show_preview(payload)
                elif kind == "done":
                    self.busy = False
                    self._toggle_buttons(True)
                    if payload.get("ok"):
                        self.status.config(text="完成", fg=C_OK)
                    else:
                        self.status.config(text="未完成（见日志）", fg=C_ERR)
        except queue.Empty:
            pass
        self.root.after(120, self._poll_queue)

    def _log(self, msg, tag=None):
        if tag is None:
            if msg.startswith(("✔", "  [球场] 检测成功")):
                tag = "ok"
            elif msg.startswith("✗"):
                tag = "err"
            elif "未检测" in msg or "降级" in msg or "未形成" in msg:
                tag = "warn"
            elif msg.startswith("  "):
                tag = "dim"
            else:
                tag = "info"
        self.log_box.insert("end", msg + "\n", tag)
        self.log_box.see("end")

    def _show_preview(self, path):
        if not (Image and os.path.exists(path)):
            return
        try:
            img = Image.open(path)
            box = (self.preview.winfo_width() or 440, self.preview.winfo_height() or 420)
            img.thumbnail((max(box[0] - 12, 200), max(box[1] - 12, 200)))
            self._preview_img = ImageTk.PhotoImage(img)
            self.preview.config(image=self._preview_img, text="")
        except Exception as e:
            self._log(f"预览加载失败：{e}", "warn")


def main():
    root = tk.Tk()
    app = HawkEyeApp(root)
    try:
        root.iconbitmap(default="")
    except Exception:
        pass
    root.mainloop()


if __name__ == "__main__":
    main()
