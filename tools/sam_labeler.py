#!/usr/bin/env python3
import argparse, os, sys
from pathlib import Path
import cv2
import numpy as np

try:
    from segment_anything import SamPredictor, sam_model_registry
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False

def load_sam(ckpt):
    if not SAM_AVAILABLE or not os.path.exists(ckpt):
        print(f"[warn] SAM 없음: {ckpt}")
        return None
    import torch
    print(f"[info] SAM 로딩: {ckpt}")
    sam = sam_model_registry["vit_b"](checkpoint=ckpt)
    sam.to("cuda" if torch.cuda.is_available() else "cpu")
    pred = SamPredictor(sam)
    print("[info] SAM 로딩 완료")
    return pred

def save_yolo(lp, bboxes, W, H, cid):
    lines = []
    for x1,y1,x2,y2 in bboxes:
        cx=((x1+x2)/2)/W; cy=((y1+y2)/2)/H
        bw=(x2-x1)/W; bh=(y2-y1)/H
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    Path(lp).write_text("\n".join(lines))

def load_yolo(lp, W, H):
    bboxes=[]
    if not Path(lp).exists(): return bboxes
    for line in Path(lp).read_text().strip().splitlines():
        p=line.split()
        if len(p)<5: continue
        _,cx,cy,bw,bh=map(float,p)
        x1=int((cx-bw/2)*W); y1=int((cy-bh/2)*H)
        x2=int((cx+bw/2)*W); y2=int((cy+bh/2)*H)
        bboxes.append((x1,y1,x2,y2))
    return bboxes

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", default=None)
    parser.add_argument("--sam-ckpt", default=os.path.expanduser("~/sj/sam_vit_b_01ec64.pth"))
    parser.add_argument("--class-id", type=int, default=0)
    parser.add_argument("--class-name", default="box")
    parser.add_argument("--negative-only", action="store_true")
    args=parser.parse_args()

    images_dir=Path(args.images)
    label_dir=Path(args.labels) if args.labels else images_dir
    label_dir.mkdir(parents=True, exist_ok=True)

    exts={".jpg",".jpeg",".png",".bmp"}
    imgs=sorted([p for p in images_dir.iterdir() if p.suffix.lower() in exts])
    print(f"[info] 이미지 {len(imgs)}장")

    if args.negative_only:
        for p in imgs:
            lp=label_dir/(p.stem+".txt")
            if not lp.exists(): lp.write_text("")
        print(f"[done] {len(imgs)}장 빈 라벨 생성 완료")
        return

    pred=load_sam(args.sam_ckpt)

    idx=0
    bboxes=[]
    drawing=False
    draw_start=None
    sam_set=False
    img_orig=None

    def load_img(i):
        nonlocal img_orig, bboxes, sam_set
        p=imgs[i]
        img_orig=cv2.imread(str(p))
        H,W=img_orig.shape[:2]
        lp=label_dir/(p.stem+".txt")
        bboxes=load_yolo(lp,W,H)
        sam_set=False
        if pred is not None:
            rgb=cv2.cvtColor(img_orig,cv2.COLOR_BGR2RGB)
            pred.set_image(rgb)
            sam_set=True
        print(f"[{i+1}/{len(imgs)}] {p.name}  기존bbox:{len(bboxes)}")

    def render():
        if img_orig is None: return np.zeros((480,640,3),dtype=np.uint8)
        d=img_orig.copy()
        H,W=d.shape[:2]
        for x1,y1,x2,y2 in bboxes:
            cv2.rectangle(d,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(d,args.class_name,(x1,max(y1-5,15)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
        info=f"[{idx+1}/{len(imgs)}] {imgs[idx].name}  bbox:{len(bboxes)}"
        guide="Ctrl+클릭=SAM자동  드래그=수동  s=저장+다음  d=neg+다음  a=이전  z=취소  q=종료"
        cv2.putText(d,info,(8,24),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,220,255),2)
        cv2.putText(d,guide,(8,46),cv2.FONT_HERSHEY_SIMPLEX,0.38,(200,200,200),1)
        return d

    def mouse_cb(event,x,y,flags,param):
        nonlocal bboxes, drawing, draw_start
        ctrl=bool(flags & cv2.EVENT_FLAG_CTRLKEY)
        H,W=img_orig.shape[:2]
        if event==cv2.EVENT_LBUTTONDOWN:
            if ctrl and pred is not None and sam_set:
                masks,scores,_=pred.predict(
                    point_coords=np.array([[x,y]]),
                    point_labels=np.array([1]),
                    multimask_output=True)
                best=masks[np.argmax(scores)]
                ys,xs=np.where(best)
                if len(xs)>0:
                    bboxes.append((int(xs.min()),int(ys.min()),int(xs.max()),int(ys.max())))
                    print(f"[SAM] ({x},{y}) → bbox추가")
            else:
                drawing=True; draw_start=(x,y)
        elif event==cv2.EVENT_LBUTTONUP:
            if drawing and draw_start:
                x1,y1_=min(draw_start[0],x),min(draw_start[1],y)
                x2,y2_=max(draw_start[0],x),max(draw_start[1],y)
                if (x2-x1)>5 and (y2_-y1_)>5:
                    bboxes.append((x1,y1_,x2,y2_))
                drawing=False; draw_start=None

    load_img(idx)
    cv2.namedWindow("SAM Labeler",cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("SAM Labeler",mouse_cb)

    while True:
        cv2.imshow("SAM Labeler",render())
        key=cv2.waitKey(20)&0xFF
        if key==ord('q'): break
        elif key==ord('s'):
            H,W=img_orig.shape[:2]
            save_yolo(label_dir/(imgs[idx].stem+".txt"),bboxes,W,H,args.class_id)
            print(f"[save] {imgs[idx].name} → {len(bboxes)}개")
            idx=min(idx+1,len(imgs)-1); load_img(idx)
        elif key==ord('d'):
            (label_dir/(imgs[idx].stem+".txt")).write_text("")
            print(f"[neg] {imgs[idx].name}")
            idx=min(idx+1,len(imgs)-1); load_img(idx)
        elif key==ord('a'):
            idx=max(idx-1,0); load_img(idx)
        elif key==ord('z'):
            if bboxes: bboxes.pop()

    cv2.destroyAllWindows()
    print("[done] 라벨링 종료")

if __name__=="__main__":
    main()
