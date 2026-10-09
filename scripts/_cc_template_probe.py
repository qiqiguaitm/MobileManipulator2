import cv2,numpy as np,time,json
cv2.setNumThreads(2)
d=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
b=cv2.aruco.CharucoBoard((7,5),1.,.737,d).generateImage((350,250))
paths=['20260923_100002_153536','20260923_101553_457262']
for name in paths:
 im=cv2.imread('log/board_approach/'+name+'/stage_0/rgb_0.png',0)
 out=[];start=time.monotonic()
 for width in np.geomspace(45,180,18).astype(int):
  for aspect in (.85,1.,1.15):
   h=int(width*5/7*aspect);a=cv2.resize(b,(width,h),interpolation=cv2.INTER_AREA)
   for angle in (-.3,-.15,0,.15,.3):
    m=np.array([[1.,0.,1.],[angle,1.,1.+max(0.,-angle*width)]])
    w2=width+2;h2=int(h+abs(angle)*width)+2
    t=cv2.warpAffine(a,m,(w2,h2),borderValue=255)
    r=cv2.matchTemplate(im,t,cv2.TM_CCOEFF_NORMED);_,score,_,pos=cv2.minMaxLoc(r)
    out.append((score,pos,width,aspect,angle))
 print(name,'seconds',time.monotonic()-start,sorted(out,reverse=True)[:8],flush=True)
