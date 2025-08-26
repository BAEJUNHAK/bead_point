# 추론(폴더/단일)→CSV/PNG
import os, glob, argparse, torch, numpy as np, cv2
from src.models import DWUNetTiny
from src.utils.viz import save_overlay
from src.losses.heatmap_utils import heatmaps_to_coords_argmax

def infer_folder(img_dir, ckpt, out_dir, size=256):  # 기본값을 256으로 변경
    os.makedirs(out_dir, exist_ok=True)
    # 모델 로드 - base=32로 변경 (훈련과 동일)
    net = DWUNetTiny(in_ch=3, base=48, out_ch=2)  # base=24 → 32로 변경
    state = torch.load(ckpt, map_location="cpu")
    # Lightning ckpt인 경우 state_dict 키 전처리 필요할 수 있음
    sd = state["state_dict"] if "state_dict" in state else state
    # 'net.' 접두어 제거
    sd = {k.replace("net.",""): v for k,v in sd.items() if "net." in k or k in net.state_dict()}
    net.load_state_dict(sd)
    net.eval()

    imgs = sorted(glob.glob(os.path.join(img_dir,"*.png")) +
                  glob.glob(os.path.join(img_dir,"*.jpg")) +
                  glob.glob(os.path.join(img_dir,"*.jpeg")))
    for p in imgs:
        rgb = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        h0,w0 = rgb.shape[:2]
        rgb_r = cv2.resize(rgb, (size,size), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(rgb_r.transpose(2,0,1)).float()/255.0
        with torch.no_grad():
            y = net(x.unsqueeze(0))
            prob = torch.sigmoid(y)
            pts = heatmaps_to_coords_argmax(prob).cpu().numpy()[0].astype(np.float32)  # [2,2]
        # 리사이즈 역변환
        sx, sy = w0/size, h0/size
        pts[:,0] *= sx; pts[:,1] *= sy
        # 저장
        base = os.path.splitext(os.path.basename(p))[0]
        np.savetxt(os.path.join(out_dir, base + "_pred.csv"), pts, delimiter=",", fmt="%.2f")
        save_overlay(os.path.join(out_dir, base + "_overlay.png"), rgb, {"pred": pts})

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_dir", required=True)
    ap.add_argument("--ckpt", required=True)  # train_img.py가 만든 best.ckpt or best.ckpt의 state_dict
    ap.add_argument("--out_dir", default="runs/infer")
    ap.add_argument("--size", type=int, default=256)  # 기본값을 256으로 변경
    args = ap.parse_args()
    infer_folder(args.img_dir, args.ckpt, args.out_dir, size=args.size)
