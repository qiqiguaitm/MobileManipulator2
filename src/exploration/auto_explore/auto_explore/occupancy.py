"""Ray-integrated voxel occupancy and timestamped PGO/LIO pose corrections."""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def transform(position, quaternion):
    t=np.eye(4);t[:3,:3]=Rotation.from_quat(quaternion).as_matrix();t[:3,3]=position;return t


def correction_at(stamp, anchors):
    """Interpolate historical corrections; latest unoptimized tail uses last anchor."""
    if not anchors:return np.eye(4)
    times=np.asarray([a[0] for a in anchors]);i=int(np.searchsorted(times,stamp,side='right'))
    if i==0:return anchors[0][1]
    if i==len(anchors):return anchors[-1][1]
    t0,c0=anchors[i-1];t1,c1=anchors[i];u=(stamp-t0)/(t1-t0)
    c=np.eye(4);c[:3,3]=(1-u)*c0[:3,3]+u*c1[:3,3]
    c[:3,:3]=Slerp([0,1],Rotation.from_matrix([c0[:3,:3],c1[:3,:3]]))([u]).as_matrix()[0]
    return c


def corrections_changed(old, new, tolerance=1e-6):
    """New keyframes alone do not require replay when the correction is unchanged.

    Compare at the union of piecewise interpolation knots. The tiny tolerance
    removes floating point noise, not actual localization corrections.
    """
    if not old:return bool(new)
    if not new:return True
    for t in sorted({t for t,_ in old+new}):
        a,b=correction_at(t,old),correction_at(t,new)
        if np.linalg.norm(a[:3,3]-b[:3,3])>tolerance:return True
        if Rotation.from_matrix(a[:3,:3].T@b[:3,:3]).magnitude()>tolerance:return True
    return False


class VoxelMap:
    def __init__(self,bounds,resolution=.05,zmin=.04,zmax=1.2):
        self.resolution=resolution;self.bounds=list(bounds);self.origin=np.array([bounds[0],bounds[1],zmin])
        self.shape=(int(np.ceil((bounds[3]-bounds[1])/resolution)),int(np.ceil((bounds[2]-bounds[0])/resolution)),int(np.ceil((zmax-zmin)/resolution)))
        self.logodds=np.zeros(self.shape,dtype=np.float32);self.seen=np.zeros(self.shape,dtype=bool)
    def ids(self,points):
        q=np.floor((points-self.origin)/self.resolution).astype(int)
        valid=(q[:,0]>=0)&(q[:,0]<self.shape[1])&(q[:,1]>=0)&(q[:,1]<self.shape[0])&(q[:,2]>=0)&(q[:,2]<self.shape[2])
        q=q[valid];return np.unique((q[:,1]*self.shape[1]+q[:,0])*self.shape[2]+q[:,2])
    def integrate(self,points,sensor_origin):
        p=np.asarray(points,float);o=np.asarray(sensor_origin,float)
        p=p[np.isfinite(p).all(axis=1)];v=p-o;dist=np.linalg.norm(v,axis=1)
        valid=(dist>=.2)&(dist<=8.);p=p[valid];v=v[valid];dist=dist[valid]
        if not len(p):return
        # Voxelized endpoints reduce repeated rays without inventing returns.
        _,indices=np.unique(np.floor(p/self.resolution).astype(int),axis=0,return_index=True)
        p=p[indices];v=v[indices];dist=dist[indices]
        hits=self.ids(p);misses=[]
        for begin in range(0,len(p),256):
            vv=v[begin:begin+256];dd=dist[begin:begin+256]
            steps=np.arange(0,float(dd.max()),self.resolution*.5)
            valid=steps[None,:] < (dd[:,None]-self.resolution*.5)
            rays=o+vv[:,None,:]*(steps[None,:,None]/dd[:,None,None])
            misses.append(self.ids(rays[valid]))
        misses=np.setdiff1d(np.unique(np.concatenate(misses)),hits,assume_unique=True)
        log=self.logodds.ravel();seen=self.seen.ravel()
        log[misses]=np.maximum(-2.,log[misses]-.4);log[hits]=np.minimum(3.5,log[hits]+.85)
        seen[misses]=True;seen[hits]=True
    def grid(self):
        occupied=np.any(self.seen&(self.logodds>=.7),axis=2)
        free=np.any(self.seen&(self.logodds<0),axis=2)
        out=np.full(self.shape[:2],-1,dtype=np.int8);out[free]=0;out[occupied]=100
        return out
