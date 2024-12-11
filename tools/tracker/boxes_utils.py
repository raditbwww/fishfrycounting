import numpy as np
import math

def iou_batch(bb_test, bb_gt):
  """
  From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
  bb_test: dets
  bb_gt: trks
  """
  bb_gt = np.expand_dims(bb_gt, 0)
  bb_test = np.expand_dims(bb_test, 1)
  xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
  yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
  xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
  yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
    + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
  return(o)

def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio with shape (4,1)
  """
  w = bbox[2] - bbox[0]
  h = bbox[3] - bbox[1]
  x = bbox[0] + w/2.
  y = bbox[1] + h/2.
  s = w * h    #scale is just area
  r = w / float(h)
  return np.array([x, y, s, r]).reshape((4, 1))

def convert_bbox_to_z2(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio with shape (1,4)
  """
  w = bbox[2] - bbox[0]
  h = bbox[3] - bbox[1]
  x = bbox[0] + w/2.
  y = bbox[1] + h/2.
  s = w * h    #scale is just area
  r = w / float(h)
  return np.array([x, y, s, r]).reshape((1, 4))


def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  w = np.sqrt(x[2] * x[3])
  h = x[2] / w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))
  
def L_diagonal(x1min,x2min,x1max,x2max,y1min,y2min,y1max,y2max):
  """
  L is diagonal distance between the smallest outer rectangle of two bounding boxes
  """
  xmin = min(x1min,x2min)
  xmax = max(x1max,x2max)
  ymin = min(y1min,y2min)
  ymax = max(y1max,y2max)

  L = (xmax-xmin)**2 + (ymax-ymin)**2
  L = math.sqrt(L)
  return L



def dij_distance(dets, trks):
  """
  Dij distance is euclidean distance between two center point of objects
  dets/trks is in [x1,y1,x2,y2,score]
  """
  # Check if dets or trks are empty and return appropriate empty array shapes
  if dets.shape[0] == 0:
    return np.empty((0, len(trks)),dtype=int)
  xdetmin = dets[...,0]
  xtrkmin = trks[...,0]
  ydetmax = dets[...,1]
  ytrkmax = trks[...,1]
  xdetmax = dets[...,2]
  xtrkmax = trks[...,2]
  ydetmin = dets[...,3]
  ytrkmin = trks[...,3]

  dets = dets[:, :-1]
  dets_cp = []
  for det in dets:
    det = convert_bbox_to_z2(det)
    dets_cp.append(det)
  dets_cp = np.array(dets_cp)
  trks = trks[:, :-1]
  trks_cp = []
  for trk in trks:
    trk = convert_bbox_to_z2(trk)
    trks_cp.append(trk)
  trks_cp = np.array(trks_cp)


  x1 = dets_cp[...,0]
  y1 = dets_cp[...,1]
  x2 = trks_cp[...,0]
  y2 = trks_cp[...,1]

  dij_matrix = np.zeros([len(x1),len(x2)])
  for i in range(len(x1)):
    for j in range(len(x2)):
      L = L_diagonal(xdetmin[i],xtrkmin[j],xdetmax[i],xtrkmax[j],ydetmin[i],ytrkmin[j],ydetmax[i],ytrkmax[j])
      dij_matrix[i][j] = 1 - (((x1[i,0]-x2[j,0])**2 + (y1[i,0]-y2[j,0])**2)/(L**2))

  return dij_matrix


def DIOU_2(iou_matrix, dij_matrix):
  """
  DIOU_2 implementation
  """
  diou_2 = (iou_matrix + dij_matrix)/2
  return diou_2

def divide_dets_byscore(dets, conf_score):
  """
  divide the detections result to high-score and low-score
  """
  if dets.shape[0]==0:
    return np.empty((0, 5),dtype=int),np.empty((0, 5),dtype=int)

  dets = dets.tolist()
  highscore_dets = []
  lowscore_dets = []
  for det in dets:
    if det[-1] >= conf_score:
      highscore_dets.append(det)  # Add to high-score list
    else:
      lowscore_dets.append(det)  # Add to low-score list
  if len(highscore_dets)==0:
    highscore_dets = np.empty((0,5))
  highscore_dets = np.array(highscore_dets)
  if len(lowscore_dets)==0:
    lowscore_dets = np.empty((0,5))
  lowscore_dets = np.array(lowscore_dets)
  return highscore_dets, lowscore_dets