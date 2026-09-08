import os,sys,json,numpy as np
R=os.path.expanduser("~/grasp/contact_graspnet_pytorch")
sys.path.insert(0,R); sys.path.insert(0,os.path.join(R,"Pointnet_Pointnet2_pytorch")); os.chdir(R)
import torch
from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator
from contact_graspnet_pytorch import config_utils
from contact_graspnet_pytorch.checkpoints import CheckpointIO
cfg=config_utils.load_config(os.path.join(R,"checkpoints","contact_graspnet"),batch_size=1)
est=GraspEstimator(cfg); CheckpointIO(checkpoint_dir=os.path.join(R,"checkpoints","contact_graspnet","checkpoints"),model=est.model).load("model.pt"); est.model.eval()
d=np.load(os.path.expanduser("~/grasp/isaac_render2_cgn.npz"),allow_pickle=True)
depth=d["depth_m"].astype(np.float32); k=d["K"].reshape(-1); K=np.array([[k[0],0,k[2]],[0,k[1],k[3]],[0,0,1]])
seg=d["seg"].astype(np.int32); W_T_cam=np.array(d["W_T_cam"]); box_w=np.array(d["box_world"])
pcf,pcs,_=est.extract_point_clouds(depth,K,segmap=seg,z_range=[0.2,1.2])
print("seg box pts",pcs[1].shape[0],"bbox cm",np.round((pcs[1].max(0)-pcs[1].min(0))*100,1))
with torch.no_grad():
    g,s,c,o=est.predict_scene_grasps(pcf,pc_segments=pcs,local_regions=True,filter_grasps=True)
G=np.asarray(g[1]).reshape(-1,4,4); S=np.asarray(s[1]).reshape(-1); O=np.asarray(o[1]).reshape(-1)
print(len(G),"grasps  score %.3f~%.3f"%(S.max(),S.min()))
flip=np.diag([1.,-1.,-1.,1.]); WT=W_T_cam@flip
def q_xyzw(Rm):
    t=np.trace(Rm)
    if t>0: s=np.sqrt(t+1)*2; w=.25*s;x=(Rm[2,1]-Rm[1,2])/s;y=(Rm[0,2]-Rm[2,0])/s;z=(Rm[1,0]-Rm[0,1])/s
    elif Rm[0,0]>Rm[1,1] and Rm[0,0]>Rm[2,2]: s=np.sqrt(1+Rm[0,0]-Rm[1,1]-Rm[2,2])*2;w=(Rm[2,1]-Rm[1,2])/s;x=.25*s;y=(Rm[0,1]+Rm[1,0])/s;z=(Rm[0,2]+Rm[2,0])/s
    elif Rm[1,1]>Rm[2,2]: s=np.sqrt(1+Rm[1,1]-Rm[0,0]-Rm[2,2])*2;w=(Rm[0,2]-Rm[2,0])/s;x=(Rm[0,1]+Rm[1,0])/s;y=.25*s;z=(Rm[1,2]+Rm[2,1])/s
    else: s=np.sqrt(1+Rm[2,2]-Rm[0,0]-Rm[1,1])*2;w=(Rm[1,0]-Rm[0,1])/s;x=(Rm[0,2]+Rm[2,0])/s;y=(Rm[1,2]+Rm[2,1])/s;z=.25*s
    v=np.array([x,y,z,w]); return (v/np.linalg.norm(v)).tolist()
out=[]
for rank,i in enumerate(np.argsort(-S)[:8]):
    gw=WT@G[i]; t=gw[:3,3]; Rm=gw[:3,:3]
    tip=t-0.1358*Rm[:,2]
    out.append(dict(rank=rank,score=float(S[i]),opening_m=float(O[i]),
        position_xyz=[round(float(x),4) for x in t],
        quat_xyzw=[round(x,5) for x in q_xyzw(Rm)],
        approach=[round(float(x),3) for x in Rm[:,2]],
        fingertip_xyz=[round(float(x),4) for x in tip],
        dbox_xy=round(float(np.hypot(tip[0]-box_w[0],tip[1]-box_w[1])),4)))
json.dump(dict(frame="base_link",box_world=box_w.tolist(),grasps=out),
          open(os.path.expanduser("~/grasp/isaac_grasps_base2.json"),"w"),indent=1)
for gg in out:
    print("  #%d s=%.3f pos=%s tip=%s dbox_xy=%.3f appr=%s"%(gg["rank"],gg["score"],gg["position_xyz"],gg["fingertip_xyz"],gg["dbox_xy"],gg["approach"]))
