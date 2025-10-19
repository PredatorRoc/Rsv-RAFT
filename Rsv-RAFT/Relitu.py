# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import os, numpy as np, cv2 as cv

# ======= 路径 =======
IMG1_PATH = r"D:\PyCharm\RAFT-master\flow_vis_stabilized\test\frame_0000-05.png"
IMG2_PATH = r"D:\PyCharm\RAFT-master\flow_vis_stabilized\test\frame_0001-05.png"

OUT_DIR   = "./flow_vis_stabilized"; os.makedirs(OUT_DIR, exist_ok=True)

# ======= 估计与物理参数 =======
MAX_SIDE  = 2048 #1320  4096*2160
ALPHA     = 0.30
PERC_LOW, PERC_HIGH = 2, 98
BORDER_MASK = 8

# GSD 与 FPS（固定为你的要求）
GSD_M_PER_PX = 0.0047346#0.00453752#0.10
FPS          = 30.0
DELTA_T_SEC  = 1.0 / FPS

# 统一色标上界（m/s）。跨时序对比可设固定值；None=按分位数自适应
VMAX_FIXED_MS = None

# ======= 可视化风格旋钮（调这里就行） =======
# 矢量箭头（quiver）
QUIVER_STRIDE            = 20     # 箭头网格间距（像素）→ 增大更稀疏，如 20/24
QUIVER_TARGET_LEN_PX     = 22     # 自适应目标箭头长度（px）→ 减小更短，如 18
QUIVER_Q                 = 0.98   # 用 q 分位估计自适应缩放
QUIVER_MIN_MAG_PX        = 0.01   # 低于该位移不画箭头
QUIVER_CLAMP_LEN_PX      = (5, 48)# 单个箭头长度限制（px）
QUIVER_THICKNESS_SCALE   = 0.6    # 箭头粗细缩放（<1 更细；>1 更粗）
QUIVER_TIP_LEN           = 0.30   # 箭头三角形比例

# 流线（含箭头）
STREAM_SEED_STRIDE       = 40     # 流线种子间距 → 增大更稀疏，如 36/40
STREAM_STEP_PX           = 1.0
STREAM_N_STEPS           = 100    # 步数；不影响密度，影响长度
STREAM_ARROW_EVERY       = 12     # 每隔多少个折点画一个箭头 → 增大更稀疏
STREAM_THICKNESS_SCALE   = 0.6    # 流线粗细缩放
STREAM_TIP_LEN           = 0.30

# ======= 工具函数 =======
def imread_rgb(p):
    bgr = cv.imread(p); assert bgr is not None, f"读不到文件: {p}"
    return cv.cvtColor(bgr, cv.COLOR_BGR2RGB)

def resize_keep(img, max_side=None):
    if not max_side or max(img.shape[:2]) <= max_side: return img
    h,w = img.shape[:2]
    if h>=w: nh,nw = max_side, int(w*max_side/h)
    else:    nw,nh = max_side, int(h*max_side/w)
    return cv.resize(img, (nw,nh), interpolation=cv.INTER_AREA)

def estimate_homography(img1_gray, img2_gray):
    orb = cv.ORB_create(5000)
    k1, d1 = orb.detectAndCompute(img1_gray, None)
    k2, d2 = orb.detectAndCompute(img2_gray, None)
    if d1 is None or d2 is None: return None
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(d1, d2, k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)
    if len(good) < 8: return None
    pts1 = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    pts2 = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    H, _ = cv.findHomography(pts2, pts1, cv.RANSAC, 3.0)
    return H

def robust_vmin_vmax(arr, mask, p1=2, p2=98):
    vals = arr[mask>0]
    if vals.size==0: return 0.0, 1.0
    vmin = float(np.percentile(vals, p1))
    vmax = float(np.percentile(vals, p2))
    if vmin==vmax: vmax = vmin + 1e-6
    return vmin, vmax

def make_masks(h,w, roi_poly=None, border=BORDER_MASK):
    mask = np.ones((h,w), np.uint8)
    if border>0:
        mask[:border,:]=0; mask[-border:,:]=0; mask[:,:border]=0; mask[:,-border:]=0
    if roi_poly is not None and len(roi_poly)>=3:
        roi = np.zeros_like(mask)
        cv.fillPoly(roi, [np.array(roi_poly,np.int32)], 1)
        mask = (mask & roi).astype(np.uint8)
    return mask

def write_flo(path, flow):
    TAG = np.array([202021.25], np.float32)
    h, w = flow.shape[:2]
    with open(path, 'wb') as f:
        TAG.tofile(f)
        np.array(w, dtype=np.int32).tofile(f)
        np.array(h, dtype=np.int32).tofile(f)
        tmp = np.empty((h, w*2), np.float32)
        tmp[:, 0::2] = flow[..., 0]
        tmp[:, 1::2] = flow[..., 1]
        tmp.tofile(f)

# ======= 可视化（热力图 + 右侧色标） =======
def colorize_magnitude(mag_ms, mask, vfixed=None):
    if vfixed is None:
        vmin, vmax = robust_vmin_vmax(mag_ms, mask, PERC_LOW, PERC_HIGH)
    else:
        vmin, vmax = 0.0, float(vfixed)
    norm = (np.clip(mag_ms, vmin, vmax) - vmin) / (vmax - vmin + 1e-12)
    gray = (norm*255).astype(np.uint8)
    try:  heat = cv.applyColorMap(gray, cv.COLORMAP_TURBO)
    except: heat = cv.applyColorMap(gray, cv.COLORMAP_JET)
    heat_rgb = cv.cvtColor(heat, cv.COLOR_BGR2RGB)
    heat_rgb[mask==0] = 0
    return heat_rgb, (vmin, vmax)

def overlay_heat_on_image(base_rgb, heat_rgb, mask, alpha=0.55):
    over = base_rgb.copy()
    over[mask>0] = (alpha*heat_rgb[mask>0] + (1-alpha)*base_rgb[mask>0]).astype(np.uint8)
    return over

# def make_right_colorbar_panel(h, vmin, vmax, label="speed (m s^-1)", panel_w=160, margin=16, n_ticks=5):
#     """
#     右侧独立色条面板：
#     - 梯度：上端=高值（红），下端=低值（蓝）
#     - 刻度与数字在色条左侧，自下而上为小→大，避免数字前的“—”被误读成负号
#     """
#     panel = np.zeros((h, panel_w, 3), dtype=np.uint8)
#
#     # 色条尺寸与位置
#     bar_w = 22
#     bar_h = int(h * 0.85)
#     y0 = (h - bar_h) // 2
#     x0 = margin
#
#     # 生成竖直渐变：顶端=255（高值），底端=0（低值）
#     grad = np.linspace(255, 0, bar_h).astype(np.uint8).reshape(bar_h, 1)
#     cm = cv.applyColorMap(grad, cv.COLORMAP_TURBO) if hasattr(cv, "COLORMAP_TURBO") else cv.applyColorMap(grad, cv.COLORMAP_JET)
#     cm = cv.cvtColor(cm, cv.COLOR_BGR2RGB)
#
#     # 贴到面板上
#     panel[y0:y0+bar_h, x0:x0+bar_w] = cv.resize(cm, (bar_w, bar_h), interpolation=cv.INTER_NEAREST)
#
#     # 刻度：自下而上，数值线性从 vmin → vmax
#     # frac_top_to_bot: 0 在顶端、1 在底端；为写刻度方便，再得到自下而上的位置
#     fracs_bottom_to_top = np.linspace(0.0, 1.0, n_ticks)  # 0=底，1=顶（阅读顺序更符合直觉）
#     for fb in fracs_bottom_to_top:
#         # y: 面板坐标里，0=顶；因此自下而上要映射为 y = y0 + (1 - fb)*bar_h
#         y = int(y0 + (1.0 - fb) * bar_h)
#         val = vmin + fb * (vmax - vmin)  # 数值从小到大
#
#         # 刻度画在色条左侧，长度6像素
#         cv.line(panel, (x0 - 6, y), (x0 - 1, y), (255, 255, 255), 1, cv.LINE_AA)
#
#         # 数字放在刻度左侧，右对齐，避免“—数字”的错觉
#         txt = f"{val:.2f}"
#         (tw, th), _ = cv.getTextSize(txt, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
#         cv.putText(panel, txt, (x0 - 8 - tw, y + 4), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv.LINE_AA)
#
#     # 标题放在色条上方
#     cv.putText(panel, label, (x0, y0 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv.LINE_AA)
#     return panel
def make_right_colorbar_panel(
    h, vmin, vmax, label="speed (m s^-1)",
    panel_w=180, margin=16, n_ticks=5,
    tick_len_px=6, gap_tick_to_text=10
):
    """
    右侧独立色条面板：
    - 白色背景，黑色刻度与文字
    - 渐变条：上 = 高值（红），下 = 低值（蓝）
    - 数字在色条右侧；刻度用竖线，避免被误读为“负号”
    """
    # 白色背景
    panel = np.full((h, panel_w, 3), 255, dtype=np.uint8)

    # 色条几何
    bar_w = 22
    bar_h = int(h * 0.85)
    y0 = (h - bar_h) // 2
    x0 = margin

    # 生成渐变（上高下低）
    grad = np.linspace(255, 0, bar_h).astype(np.uint8).reshape(bar_h, 1)
    cm = cv.applyColorMap(grad, cv.COLORMAP_TURBO) if hasattr(cv, "COLORMAP_TURBO") else cv.applyColorMap(grad, cv.COLORMAP_JET)
    cm = cv.cvtColor(cm, cv.COLOR_BGR2RGB)
    panel[y0:y0+bar_h, x0:x0+bar_w] = cv.resize(cm, (bar_w, bar_h), interpolation=cv.INTER_NEAREST)

    # 色条右缘分隔竖线
    x_sep = x0 + bar_w + 2
    cv.line(panel, (x_sep, y0), (x_sep, y0+bar_h), (60,60,60), 1, cv.LINE_AA)

    # 刻度（自下而上 vmin→vmax），竖线 + 右侧黑色数字
    fracs_bottom_to_top = np.linspace(0.0, 1.0, n_ticks)  # 0=底，1=顶
    for fb in fracs_bottom_to_top:
        y = int(y0 + (1.0 - fb) * bar_h)       # 面板坐标：0 在顶
        val = vmin + fb * (vmax - vmin)        # 数值从小到大

        # 竖向刻度
        cv.line(panel, (x_sep + 1, y - tick_len_px//2), (x_sep + 1, y + tick_len_px//2), (0,0,0), 1, cv.LINE_AA)

        # 右侧黑色文字
        txt = f"{val:.2f}"
        x_txt = x_sep + 1 + gap_tick_to_text
        cv.putText(panel, txt, (x_txt, y + 4), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv.LINE_AA)

    # 标题（黑色）
    cv.putText(panel, label, (x0, y0 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 1, cv.LINE_AA)
    return panel


def concat_with_right_panel(image_rgb, panel_rgb):
    h = image_rgb.shape[0]
    if panel_rgb.shape[0] != h:
        panel_rgb = cv.resize(panel_rgb, (panel_rgb.shape[1], h), interpolation=cv.INTER_AREA)
    return np.concatenate([image_rgb, panel_rgb], axis=1)

# ======= 箭头自适应 & 绘制 =======
def auto_quiver_scale(flow_px, mask, target_len_px=28, q=0.95, eps=1e-6):
    mag = np.sqrt(flow_px[...,0]**2 + flow_px[...,1]**2)
    vals = mag[mask>0]
    if vals.size == 0:
        return 2.0
    ref = float(np.percentile(vals, q*100.0))
    if ref < eps:
        ref = float(vals.mean() + eps)
    return float(target_len_px / max(ref, eps))

def draw_quiver(image_rgb, flow_px, mask,
                stride=QUIVER_STRIDE, scale_px=None, target_len_px=QUIVER_TARGET_LEN_PX,
                q=QUIVER_Q, min_mag_px=QUIVER_MIN_MAG_PX, clamp_len_px=QUIVER_CLAMP_LEN_PX):
    H, W = mask.shape
    vis = image_rgb.copy()
    flow_s = cv.GaussianBlur(flow_px, (0,0), 1.0)

    if scale_px is None:
        scale_px = auto_quiver_scale(flow_s, mask, target_len_px=target_len_px, q=q)

    base = max(H, W)
    thick_b = max(1, int(round(base * 0.003 * QUIVER_THICKNESS_SCALE)))  # 黑描边
    thick_w = max(1, thick_b - 1)                                       # 白箭头
    tip_len  = QUIVER_TIP_LEN

    for y in range(stride//2, H, stride):
        for x in range(stride//2, W, stride):
            if mask[y, x] == 0:
                continue
            dx, dy = float(flow_s[y, x, 0]), float(flow_s[y, x, 1])
            mag = (dx*dx + dy*dy) ** 0.5
            if mag < min_mag_px:
                continue
            L = mag * scale_px
            L = min(max(L, clamp_len_px[0]), clamp_len_px[1])
            s_local = 0.0 if mag == 0.0 else (L / mag)

            x2 = int(round(x + dx * s_local))
            y2 = int(round(y + dy * s_local))

            cv.arrowedLine(vis, (x, y), (x2, y2), (0, 0, 0), max(1, thick_b), cv.LINE_AA, 0, tip_len)
            cv.arrowedLine(vis, (x, y), (x2, y2), (255, 255, 255), max(1, thick_w), cv.LINE_AA, 0, tip_len)
    return vis

# ======= 流线 + 箭头 =======
def _bilinear_sample(vec, x, y):
    H, W, _ = vec.shape
    if x<0 or x>W-1 or y<0 or y>H-1: return 0.0, 0.0
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = min(x0+1, W-1), min(y0+1, H-1)
    fx, fy = x - x0, y - y0
    v00 = vec[y0, x0]; v10 = vec[y0, x1]; v01 = vec[y1, x0]; v11 = vec[y1, x1]
    vx = (1-fx)*(1-fy)*v00[0] + fx*(1-fy)*v10[0] + (1-fx)*fy*v01[0] + fx*fy*v11[0]
    vy = (1-fx)*(1-fy)*v00[1] + fx*(1-fy)*v10[1] + (1-fx)*fy*v01[1] + fx*fy*v11[1]
    return vx, vy

def _rk4_step(vec, x, y, h):
    k1 = _bilinear_sample(vec, x, y)
    k2 = _bilinear_sample(vec, x + 0.5*h*k1[0], y + 0.5*h*k1[1])
    k3 = _bilinear_sample(vec, x + 0.5*h*k2[0], y + 0.5*h*k2[1])
    k4 = _bilinear_sample(vec, x + h*k3[0],     y + h*k3[1])
    vx = (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]) / 6.0
    vy = (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]) / 6.0
    return x + h*vx, y + h*vy

def draw_streamlines_with_arrows(image_rgb, flow_px, mask,
                                 seed_stride=STREAM_SEED_STRIDE, step_px=STREAM_STEP_PX,
                                 n_steps=STREAM_N_STEPS, arrow_every=STREAM_ARROW_EVERY):
    H, W = mask.shape
    vis = image_rgb.copy()

    base = max(H, W)
    thick_b = max(1, int(round(base * 0.003 * STREAM_THICKNESS_SCALE)))
    thick_w = max(1, thick_b - 1)
    tip_len  = STREAM_TIP_LEN

    flow_s = cv.GaussianBlur(flow_px, (0,0), 1.0)

    seeds = []
    for y in range(seed_stride//2, H, seed_stride):
        for x in range(seed_stride//2, W, seed_stride):
            if mask[y, x]:
                seeds.append((float(x), float(y)))

    for sx, sy in seeds:
        for direction in (+1, -1):
            x, y = sx, sy
            pts = []
            for _ in range(n_steps):
                if x < 1 or x >= W-1 or y < 1 or y >= H-1 or mask[int(y), int(x)] == 0:
                    break
                pts.append((int(round(x)), int(round(y))))
                x, y = _rk4_step(flow_s * direction, x, y, step_px)
            if len(pts) < 6:
                continue

            for i in range(1, len(pts)):
                cv.line(vis, pts[i-1], pts[i], (0, 0, 0), max(1, thick_b), cv.LINE_AA)
                cv.line(vis, pts[i-1], pts[i], (255, 255, 255), max(1, thick_w), cv.LINE_AA)

            for j in range(arrow_every, len(pts), arrow_every):
                p0 = pts[j-1]; p1 = pts[j]
                cv.arrowedLine(vis, p0, p1, (0, 0, 0), max(1, thick_b+1), cv.LINE_AA, 0, tip_len)
                cv.arrowedLine(vis, p0, p1, (255, 255, 255), max(1, thick_w),   cv.LINE_AA, 0, tip_len)
    return vis

#======= RAFT（torchvision） =======
def flow_raft(img1_rgb, img2_rgb):
    import torch
    from PIL import Image
    from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    w = Raft_Small_Weights.DEFAULT
    m = raft_small(weights=w).to(dev).eval()
    t = w.transforms()
    t1,t2 = t(Image.fromarray(img1_rgb), Image.fromarray(img2_rgb))
    t1,t2 = t1.unsqueeze(0).to(dev), t2.unsqueeze(0).to(dev)
    with torch.inference_mode():
        flow = m(t1,t2)
        if isinstance(flow,(list,tuple)): flow = flow[-1]
        flow = flow[0].detach().cpu().numpy().transpose(1,2,0)  # (Hf,Wf,2) px/帧
    H0,W0 = img1_rgb.shape[:2]
    Hf,Wf = flow.shape[:2]
    if (Hf,Wf)!=(H0,W0):
        sx,sy = W0/Wf, H0/Hf
        flow = cv.resize(flow,(W0,H0),interpolation=cv.INTER_LINEAR)
        flow[...,0]*=sx; flow[...,1]*=sy
    return flow  # 像素/帧

#use my own pth
# def _build_raft_and_transforms(variant="large", ckpt_path=None, strict=False, device="cuda"):
#     """
#     构建 RAFT (torchvision 实现)，并尝试加载自定义权重。
#     - variant: "small" 或 "large"
#     - ckpt_path: 自定义权重 .pth/.pt；若为 None/"" 则用默认预训练
#     - strict: load_state_dict 严格与否
#     返回: (model.eval().to(device), transforms)
#     """
#     import torch
#     from torchvision.models.optical_flow import (
#         raft_small, raft_large,
#         Raft_Small_Weights, Raft_Large_Weights
#     )
#
#     use_custom = bool(ckpt_path)
#
#     if variant.lower() == "large":
#         default_w = Raft_Large_Weights.DEFAULT
#         model = raft_large(weights=None if use_custom else default_w)
#     else:
#         default_w = Raft_Small_Weights.DEFAULT
#         model = raft_small(weights=None if use_custom else default_w)
#
#     transforms = default_w.transforms()   # 用同一套预处理
#     model = model.to(device).eval()
#
#     if use_custom:
#         print(f"[Info] 加载自定义 RAFT 权重: {ckpt_path}")
#         obj = torch.load(ckpt_path, map_location="cpu")
#
#         # 兼容常见保存格式
#         if isinstance(obj, dict):
#             sd = obj.get("state_dict") or obj.get("model_state_dict") or obj.get("model") or obj
#         else:
#             sd = obj
#
#         # 去常见前缀（DataParallel/Lightning 等）
#         def strip_prefix(sdict, prefixes=("module.", "model.", "raft.")):
#             out = {}
#             for k, v in sdict.items():
#                 kk = k
#                 for p in prefixes:
#                     if kk.startswith(p):
#                         kk = kk[len(p):]
#                 out[kk] = v
#             return out
#
#         sd = strip_prefix(sd)
#
#         missing, unexpected = model.load_state_dict(sd, strict=strict)
#     return model, transforms
#
#
# def flow_raft(img1_rgb, img2_rgb,
#               ckpt_path=r"D:\PyCharm\RAFT-master\raft-model\ultra+C+T\raft-chairs+things.pth",
#               variant="large",
#               strict=False):
#     """
#     用 torchvision 的 RAFT 推理；可选加载自定义权重。
#     返回 flow: (H, W, 2) 以 像素/帧 表示
#     """
#     import torch
#     from PIL import Image
#
#     dev = "cuda" if torch.cuda.is_available() else "cpu"
#     model, t = _build_raft_and_transforms(
#         variant=variant, ckpt_path=ckpt_path, strict=strict, device=dev
#     )
#
#     # 预处理
#     t1, t2 = t(Image.fromarray(img1_rgb), Image.fromarray(img2_rgb))
#     t1, t2 = t1.unsqueeze(0).to(dev), t2.unsqueeze(0).to(dev)
#
#     # 推理
#     with torch.inference_mode():
#         flow = model(t1, t2)
#         if isinstance(flow, (list, tuple)):
#             flow = flow[-1]
#         flow = flow[0].detach().cpu().numpy().transpose(1, 2, 0)  # (Hf,Wf,2) px/帧
#
#     # 若 RAFT 输出分辨率和输入不同，则插值回原图尺度并按比例缩放位移
#     H0, W0 = img1_rgb.shape[:2]
#     Hf, Wf = flow.shape[:2]
#     if (Hf, Wf) != (H0, W0):
#         sx, sy = W0 / Wf, H0 / Hf
#         flow = cv.resize(flow, (W0, H0), interpolation=cv.INTER_LINEAR)
#         flow[..., 0] *= sx
#         flow[..., 1] *= sy
#
#     return flow

# ======= 相机运动扣除 =======
def homography_flow(H, h, w):
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    pts = np.stack([xx.ravel(), yy.ravel(), np.ones_like(xx).ravel()], axis=0)
    p2 = H @ pts
    p2 /= p2[2:3, :]
    dx = (p2[0, :] - xx.ravel()).reshape(h, w)
    dy = (p2[1, :] - yy.ravel()).reshape(h, w)
    return np.stack([dx, dy], axis=-1)

# ======= 主流程 =======
def main():
    # 1) 读图 & 缩放
    img1 = imread_rgb(IMG1_PATH); img2 = imread_rgb(IMG2_PATH)
    img1 = resize_keep(img1, MAX_SIDE); img2 = resize_keep(img2, MAX_SIDE)

    # 2) 单应性（仅用于扣除相机运动）
    g1 = cv.cvtColor(img1, cv.COLOR_RGB2GRAY)
    g2 = cv.cvtColor(img2, cv.COLOR_RGB2GRAY)
    H = estimate_homography(g1, g2)
    if H is not None: print("[Info] 找到单应性，用于相机流扣除。")
    else:             print("[Warn] 未找到可靠单应性，相机运动可能残留。")

    # 3) RAFT 光流（像素/帧）
    try:
        flow_raw = flow_raft(img1, img2)
        backend = "RAFT"
    except Exception as e:
        raise RuntimeError(f"RAFT 不可用或运行失败：{e}")

    # 3.1) 去相机运动
    if H is not None:
        cam_flow = homography_flow(H, *img1.shape[:2])
        flow_px = flow_raw - cam_flow
        print("[Info] 已从光流中减去相机分量。")
    else:
        flow_px = flow_raw

    # 4) 掩膜
    h,w = img1.shape[:2]
    roi_poly = None
    mask = make_masks(h, w, roi_poly, border=BORDER_MASK)

    # 5) 速度大小（m/s）
    mag_ms = np.sqrt((flow_px[...,0]*GSD_M_PER_PX/DELTA_T_SEC)**2 +
                     (flow_px[...,1]*GSD_M_PER_PX/DELTA_T_SEC)**2)

    # 6) 热力图 + 叠加
    heat_rgb, (vmin, vmax) = colorize_magnitude(mag_ms, mask, vfixed=VMAX_FIXED_MS)
    overlay = overlay_heat_on_image(img1, heat_rgb, mask, alpha=ALPHA)

    # 7) 矢量箭头（更细、更稀疏；可在顶部旋钮改）
    quiver_img = draw_quiver(overlay, flow_px, mask)

    # 8) 流线 + 箭头（更细、更稀疏）
    stream_img = draw_streamlines_with_arrows(overlay, flow_px, mask)
    heat_quiver_img = draw_quiver(heat_rgb, flow_px, mask)  # 在热力图上叠加箭头

    # 9) 右侧独立色标
    right_panel = make_right_colorbar_panel(h, vmin, vmax, label="speed (m/s)", panel_w=160, margin=16)

    # 10) 拼接并保存
    overlay_with_bar      = concat_with_right_panel(overlay,     right_panel)
    quiver_with_bar       = concat_with_right_panel(quiver_img,  right_panel)
    stream_with_bar       = concat_with_right_panel(stream_img,  right_panel)
    heat_only_with_bar    = concat_with_right_panel(heat_rgb,    right_panel)
    heat_quiver_with_bar = concat_with_right_panel(heat_quiver_img, right_panel)  # ← 新增

    base = os.path.splitext(os.path.basename(IMG1_PATH))[0]
    cv.imwrite(os.path.join(OUT_DIR, f"{base}_{backend}_heat_rightbar.png"),            cv.cvtColor(heat_only_with_bar, cv.COLOR_RGB2BGR))
    cv.imwrite(os.path.join(OUT_DIR, f"{base}_{backend}_overlay_rightbar.png"),         cv.cvtColor(overlay_with_bar,   cv.COLOR_RGB2BGR))
    cv.imwrite(os.path.join(OUT_DIR, f"{base}_{backend}_overlay_quiver_rightbar.png"),  cv.cvtColor(quiver_with_bar,    cv.COLOR_RGB2BGR))
    cv.imwrite(os.path.join(OUT_DIR, f"{base}_{backend}_overlay_streamlines_rightbar.png"), cv.cvtColor(stream_with_bar, cv.COLOR_RGB2BGR))
    cv.imwrite(os.path.join(OUT_DIR, f"{base}_{backend}_heat_quiver_rightbar.png"),
               cv.cvtColor(heat_quiver_with_bar, cv.COLOR_RGB2BGR))

    write_flo(os.path.join(OUT_DIR, f"{base}_{backend}.flo"), flow_px)
    np.save(os.path.join(OUT_DIR, f"{base}_{backend}.npy"), flow_px)

    print(f"[OK] 后端: {backend} | 单位: m/s | 色标: {vmin:.4f} ~ {vmax:.4f}")

if __name__ == "__main__":
    main()

