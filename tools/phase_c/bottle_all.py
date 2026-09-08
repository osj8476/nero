import os,sys,json,numpy as np
R=os.path.expanduser("~/grasp/contact_graspnet_pytorch")
sys.path.insert(0,R); sys.path.insert(0,os.path.join(R,"Pointnet_Pointnet2_pytorch")); os.chdir(R)
import torch
from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator
from contact_graspnet_pytorch import config_utils
from contact_graspnet_pytorch.checkpoints import CheckpointIO
cfg=config_utils.load_config(os.path.join(R,"checkpoints","contact_graspnet"),batch_size=1)
est=GraspEstimator(cfg); CheckpointIO(checkpoint_dir=os.path.join(R,"checkpoints","contact_graspnet","checkpoints"),model=est.model).load("model.pt"); est.model.eval()
d=np.load(os.path.expanduser("~/grasp/bottle_cgn.npz"),allow_pickle=True)
depth=d["depth_m"].astype(np.float32); k=d["K"].reshape(-1); K=np.array([[k[0],0,k[2]],[0,k[1],k[3]],[0,0,1]])
seg=d["seg"].astype(np.int32); W_T_cam=np.array(d["W_T_cam"]); obj=np.array(d["obj_world"])
pcf,pcs,_=est.extract_point_clouds(depth,K,segmap=seg,z_range=[0.2,1.2])
with torch.no_grad():
    g,s,c,o=est.predict_scene_grasps(pcf,pc_segments=pcs,local_regions=True,filter_grasps=True)
G=np.asarray(g[1]).reshape(-1,4,4); S=np.asarray(s[1]).reshape(-1); O=np.asarray(o[1]).reshape(-1)
print(len(G),"grasps  score %.3f-%.3f"%(S.min(),S.max()))
flip=np.diag([1.,-1.,-1.,1.]); WT=W_T_cam@flip
def R2q(R):
    t=np.trace(R)
    if t>0: sq=np.sqrt(t+1)*2;w=.25*sq;x=(R[2,1]-R[1,2])/sq;y=(R[0,2]-R[2,0])/sq;z=(R[1,0]-R[0,1])/sq
    elif R[0,0]>R[1,1] and R[0,0]>R[2,2]: sq=np.sqrt(1+R[0,0]-R[1,1]-R[2,2])*2;w=(R[2,1]-R[1,2])/sq;x=.25*sq;y=(R[0,1]+R[1,0])/sq;z=(R[0,2]+R[2,0])/sq
    elif R[1,1]>R[2,2]: sq=np.sqrt(1+R[1,1]-R[0,0]-R[2,2])*2;w=(R[0,2]-R[2,0])/sq;x=(R[0,1]+R[1,0])/sq;y=.25*sq;z=(R[1,2]+R[2,1])/sq
    else: sq=np.sqrt(1+R[2,2]-R[0,0]-R[1,1])*2;w=(R[1,0]-R[0,1])/sq;x=(R[0,2]+R[2,0])/sq;y=(R[1,2]+R[2,1])/sq;z=.25*sq
    v=np.array([x,y,z,w]);return (v/np.linalg.norm(v)).tolist()
out=[]
for i in range(len(G)):
    gw=WT@G[i]
    out.append(dict(rank=i,score=float(S[i]),opening_m=float(O[i]),
        position_xyz=[round(float(x),4) for x in gw[:3,3]],
        quat_xyzw=[round(x,5) for x in R2q(gw[:3,:3])],
        approach=[round(float(x),3) for x in gw[:3,2]]))
json.dump(dict(frame="base_link",box_world=obj.tolist(),grasps=out),
          open(os.path.expanduser("~/grasp/bottle_grasps_all.json"),"w"),indent=1)
print("wrote",len(out),"-> bottle_grasps_all.json")
