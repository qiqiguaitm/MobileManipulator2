"""Capture-time alignment diagram; diagnostics only, no motion commands."""
import cv2
import numpy as np
from .limits import FINAL_LATERAL_TOLERANCE_M, FINAL_HEADING_TOLERANCE_DEG


def alignment_metrics(center, normal):
    center=np.asarray(center,float);normal=np.asarray(normal,float)
    if center.shape!=(3,) or normal.shape!=(3,) or not np.isfinite(np.r_[center,normal]).all() or np.linalg.norm(normal[:2])<.9:
        raise ValueError('Invalid alignment geometry')
    n=normal[:2]/np.linalg.norm(normal[:2])
    return dict(distance=float(-center[:2]@n),
                lateral=float(center[:2]@np.array([-n[1],n[0]])),
                heading_deg=float(np.degrees(np.arctan2(-n[1],-n[0]))))


def save_alignment(path, center, normal, plane_fit=None, points=None, fit_points=None, inliers=None, excluded_points=None):
    metrics=alignment_metrics(center,normal)
    center=np.asarray(center,float)[:2];normal=np.asarray(normal,float)[:2];normal/=np.linalg.norm(normal)
    canvas=np.full((720,760,3),25,np.uint8)
    scale=min(150.,340./max(np.linalg.norm(center),1.))
    def px(p):return tuple(np.rint([380-scale*p[1],610-scale*p[0]]).astype(int))
    def arrow(a,b,color):cv2.arrowedLine(canvas,px(a),px(b),color,3,tipLength=.2)
    for x in np.arange(0,5,.5):cv2.line(canvas,px([x,-3]),px([x,3]),(55,55,55),1)
    def draw_cloud(values,color,radius):
        if values is None:return
        for point in np.asarray(values,float):
            if not np.isfinite(point).all():continue
            u,v=px(point[:2])
            if 0<=u<760 and 225<=v<720:cv2.circle(canvas,(u,v),radius,color,-1)
    draw_cloud(points,(75,75,75),1)
    draw_cloud(excluded_points,(220,80,220),2)
    if fit_points is not None and inliers is not None:
        cloud=np.asarray(fit_points,float);mask=np.asarray(inliers,bool)
        if len(cloud)==len(mask):
            draw_cloud(cloud[~mask],(30,30,240),3)
            draw_cloud(cloud[mask],(30,230,30),3)
    cv2.rectangle(canvas,px([.434,.31]),px([-.375,-.31]),(180,180,180),2)
    tangent=np.array([-normal[1],normal[0]])
    cv2.line(canvas,px(center-.4*tangent),px(center+.4*tangent),(255,255,255),4)
    cv2.circle(canvas,px(center),5,(255,255,255),-1)
    arrow(center,center+.6*normal,(255,255,0))
    arrow([0,0],[.7,0],(0,255,0))
    arrow([0,0],-.7*normal,(0,165,255))
    lines=['TOP VIEW / base_link at image capture',
           'Green: robot forward   Orange: required heading',
           'Cyan: outward face normal   White: face plane',
           f'Heading error: {metrics["heading_deg"]:+.2f} deg (limit +/-{FINAL_HEADING_TOLERANCE_DEG:g})',
           f'Lateral: {metrics["lateral"]:+.3f} m (limit +/-{FINAL_LATERAL_TOLERANCE_M:.2f})',
           f'Base-to-plane: {metrics["distance"]:.3f} m']
    fit=plane_fit or {}
    if fit.get('half_to_full_angles_deg') is not None:
        lines.append('Left/right vs fitted plane: '+', '.join(f'{v:.2f} deg' for v in fit['half_to_full_angles_deg']))
    cv2.rectangle(canvas,(0,0),(760,225),(25,25,25),-1)
    for i,line in enumerate(lines):cv2.putText(canvas,line,(14,27+i*28),cv2.FONT_HERSHEY_SIMPLEX,.55,(240,240,240),1,cv2.LINE_AA)
    cv2.putText(canvas,'Points: green=inlier red=outlier magenta=door excluded grey=other',
                (12,705),cv2.FONT_HERSHEY_SIMPLEX,.49,(240,240,240),1,cv2.LINE_AA)
    if not cv2.imwrite(str(path),canvas):raise OSError('Could not save alignment diagram')
    return metrics
