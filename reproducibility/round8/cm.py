import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":6.5})
def ro(m,o): m=np.array(m); return m[np.ix_(o,o)]
D=[]
# OASIS original order Mild,Mod,Non,VM -> Non,VM,Mild,Mod
o=[2,3,0,1]
D.append(("OASIS",["ND","VMD","MD","MoD"],
 ro([[396,2,49,54],[1,36,9,5],[8,0,6664,25],[11,2,222,1161]],o),
 ro([[398,1,49,53],[1,36,9,5],[8,0,6664,25],[11,2,221,1162]],o),[6697,1396,501,51]))
o=[2,4,0,1,3]
D.append(("AD5",["N","VM","Mi","Mo","Se"],
 ro([[42,0,7,0,0],[0,138,0,15,0],[0,0,21,0,0],[0,0,0,378,91],[0,0,0,0,29]],o),
 ro([[42,0,7,0,0],[0,138,0,15,0],[0,0,21,0,0],[0,0,0,438,31],[0,0,0,2,27]],o),[21,29,49,153,469]))
D.append(("BreastMNIST",["Mal","Ben"],np.array([[28,14],[15,99]]),np.array([[32,10],[26,88]]),[42,114]))
D.append(("PneumoniaMNIST",["Norm","Pneu"],np.array([[187,47],[14,376]]),np.array([[187,47],[13,377]]),[234,390]))
D.append(("RetinaMNIST",["0","1","2","3","4"],
 np.array([[132,2,17,23,0],[13,2,17,14,0],[21,3,24,44,0],[4,1,18,45,0],[4,0,6,10,0]]),
 np.array([[131,2,25,16,0],[13,2,22,9,0],[18,3,43,28,0],[4,1,27,36,0],[4,0,7,9,0]]),[174,46,92,68,20]))
D.append(("DermaMNIST",["AKIEC","BCC","BKL","DF","MEL","NV","VASC"],
 np.array([[21,24,9,0,2,10,0],[4,66,5,2,2,22,2],[7,23,93,0,12,85,0],[0,10,0,2,0,11,0],[4,11,31,1,51,124,1],[3,14,22,2,19,1281,0],[0,1,0,0,1,14,13]]),
 np.array([[29,18,8,0,1,10,0],[9,62,4,2,2,22,2],[11,19,93,0,12,85,0],[2,8,0,2,0,11,0],[6,9,31,1,50,124,2],[4,13,22,2,19,1281,0],[0,1,0,0,1,14,13]]),[66,103,220,23,223,1341,29]))
for name,lab,b,f,sup in D:
    assert list(b.sum(1))==sup and list(f.sum(1))==sup, name
    d=f-b; print(name,"correct",np.trace(b),"->",np.trace(f), "changes into col:",d.clip(min=0).sum(0))
fig,axes=plt.subplots(3,4,figsize=(6.7,5.2),gridspec_kw=dict(wspace=0.55,hspace=0.42))
for k,(name,lab,b,f,sup) in enumerate(D):
    r,c=divmod(k,2); a1=axes[r,2*c]; a2=axes[r,2*c+1]
    n=len(lab); fs=(5.4 if n==4 else 6.2) if n<=4 else (5.6 if n==5 else 4.6)
    norm=b/b.sum(1,keepdims=True)
    a1.imshow(norm,cmap="Blues",vmin=0,vmax=1)
    for i in range(n):
        for j in range(n):
            a1.text(j,i,str(b[i,j]),ha="center",va="center",fontsize=fs,color="white" if norm[i,j]>0.55 else "black")
    d=f-b; m=max(1,np.abs(d).max())
    a2.imshow(d,cmap="RdBu",norm=TwoSlopeNorm(0,-m,m))
    for i in range(n):
        for j in range(n):
            if d[i,j]!=0: a2.text(j,i,f"{d[i,j]:+d}",ha="center",va="center",fontsize=fs,color="white" if abs(d[i,j])>0.6*m else "black")
            elif i==j: a2.text(j,i,"0",ha="center",va="center",fontsize=fs,color="0.6")
    for a,t in ((a1,f"{name}: Phase 1"),(a2,f"{name}: change ($\Delta$)")):
        a.set_xticks(range(n)); a.set_yticks(range(n))
        a.set_xticklabels(lab,rotation=90 if n>5 else 0,fontsize=5.4); a.set_yticklabels(lab,fontsize=5.4)
        a.set_title(t,fontsize=6.4,pad=3)
        a.tick_params(length=1.5,pad=1)
    a1.set_ylabel("True",fontsize=6,labelpad=1)
    a1.set_xlabel("Predicted",fontsize=6,labelpad=1); a2.set_xlabel("Predicted",fontsize=6,labelpad=1)
fig.savefig("confusion_delta.pdf",bbox_inches="tight"); fig.savefig("confusion_delta.png",dpi=200,bbox_inches="tight")
