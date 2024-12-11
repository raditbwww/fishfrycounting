import numpy as np

from boxes_utils import iou_batch, dij_distance, DIOU_2, divide_dets_byscore
from kalmanfilter import KalmanBoxTracker

# for association matching
def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))

# associate detection to trackers
# 1st stage
def associate_detections_firststage(highscore_detections,trackers,dij_threshold = 0.9):
  """
  Assigns detections to tracked object (both represented as bounding boxes) with DiJ Euclidean Distance Method
  For High Score Detections
  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers)==0): # what if there is no high-score detections? -> already handled in dij_distance
    return np.empty((0,2),dtype=int), np.arange(len(highscore_detections)), np.empty((0,5),dtype=int)

  dij_matrix = dij_distance(highscore_detections, trackers)

  if min(dij_matrix.shape) > 0:
    a = (dij_matrix > dij_threshold).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched_indices = np.stack(np.where(a), axis=1)
    else:
      matched_indices = linear_assignment(-dij_matrix)
  else:
    matched_indices = np.empty(shape=(0,2))

  unmatched_detections = []
  for d, det in enumerate(highscore_detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)


  #filter out matched with low dij value
  matches = []
  for m in matched_indices:
    if(dij_matrix[m[0], m[1]]<dij_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

# 2nd stage
def associate_detections_secondstage(lowscore_dets,trks,unmatched_trackers_prev,diou_threshold = 0.7):
  """
  Assigns detections to tracked object (both represented as bounding boxes) for low-score with DIOU_2 method
  lowscore_dets; trks; unmatched_trks: indices of trks that is not matched
  Returns 2 lists of matched and unmatched_trackers
  """
  if (len(trks)==0) or (len(unmatched_trackers_prev)==0):
    return np.empty((0,2),dtype=int), unmatched_trackers_prev

  # extract the value of unmatched_trackers_prev from trks to create unmatched_trks
  unmatched_trks = []
  for index in unmatched_trackers_prev:
    unmatched_trks.append(trks[index])
  unmatched_trks = np.array(unmatched_trks)

  # matrix calc
  iou_matrix = iou_batch(lowscore_dets,unmatched_trks)
  dij_matrix = dij_distance(lowscore_dets,unmatched_trks)
  diou_matrix = DIOU_2(iou_matrix, dij_matrix)

  if min(diou_matrix.shape) > 0:
    a = (diou_matrix > diou_threshold).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched_indices = np.stack(np.where(a), axis=1)
    else:
      matched_indices = linear_assignment(-diou_matrix)
  else:
    matched_indices = np.empty(shape=(0,2))

  unmatched_detections = []
  for d, det in enumerate(lowscore_dets):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(unmatched_trks):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low DIOU
  matches = []
  for m in matched_indices:
    if(diou_matrix[m[0], m[1]]<diou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  # replace with elements with true value
  # Replace the second element in each row of `matches` using the second element as the index for unmatched_trks
  for i in range(matches.shape[0]):
      # Use the second element in the row as the index for unmatched_trks
      index = matches[i, 1]
      matches[i, 1] = unmatched_trackers_prev[index]
  # Replace the element in unmatched_trks with the corresponding value from unmatched_trks_prev
  for i,value in enumerate(unmatched_trackers):
    unmatched_trackers[i] = unmatched_trackers_prev[value]

  return matches, np.array(unmatched_trackers)

# 3rd stage
def associate_detections_thirdstage(dets,trks,unmatched_trackers_prev,unmatched_detections_prev,diou_threshold = 0.7):
  """
  Assigns detections to tracked object (both represented as bounding boxes) for low-score with DIOU_2 method with higher threshold

  Returns 3 lists of matched and unmatched_trackers
  """
  if (len(dets)==0) or (len(trks)==0) or (len(unmatched_trackers_prev)==0) or (len(unmatched_detections_prev)==0):
    return np.empty((0,2),dtype=int), unmatched_detections_prev, unmatched_trackers_prev

  # extract the value of dets[unmatched_detections_prev] to create unmatched_trks
  unmatched_dets = []
  for index in unmatched_detections_prev:
    unmatched_dets.append(dets[index])
  unmatched_dets = np.array(unmatched_dets)

  # extract the value of unmatched_trackers_prev from trks to create unmatched_trks
  unmatched_trks = []
  for index in unmatched_trackers_prev:
    unmatched_trks.append(trks[index])
  unmatched_trks = np.array(unmatched_trks)

  # matrix calc
  iou_matrix = iou_batch(unmatched_dets,unmatched_trks)
  dij_matrix = dij_distance(unmatched_dets,unmatched_trks)
  diou_matrix = DIOU_2(iou_matrix, dij_matrix)

  if min(diou_matrix.shape) > 0:
    a = (diou_matrix > diou_threshold).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched_indices = np.stack(np.where(a), axis=1)
    else:
      matched_indices = linear_assignment(-diou_matrix)
  else:
    matched_indices = np.empty(shape=(0,2))

  unmatched_detections = []
  for d, det in enumerate(unmatched_dets):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(unmatched_trks):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low DIOU
  matches = []
  for m in matched_indices:
    if(diou_matrix[m[0], m[1]]<diou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  # replace with elements with true value for detections
  # Replace the second element in each row of `matches` using the second element as the index for unmatched_dets
  for i in range(matches.shape[0]):
      # Use the second element in the row as the index for unmatched_trks
      index = matches[i, 0]
      matches[i, 0] = unmatched_detections_prev[index]
  # Replace the element in unmatched_trks with the corresponding value from unmatched_trks_prev
  for i,value in enumerate(unmatched_detections):
    unmatched_detections[i] = unmatched_detections_prev[value]

  # replace with elements with true value for trackers
  # Replace the second element in each row of `matches` using the second element as the index for unmatched_trks
  for i in range(matches.shape[0]):
      # Use the second element in the row as the index for unmatched_trks
      index = matches[i, 1]
      matches[i, 1] = unmatched_trackers_prev[index]
  # Replace the element in unmatched_trks with the corresponding value from unmatched_trks_prev
  for i,value in enumerate(unmatched_trackers):
    unmatched_trackers[i] = unmatched_trackers_prev[value]

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

# the sort itself
class Sort(object):
  def __init__(self, max_age=1, min_hits=1, dij_threshold=0.9, diou_threshold_2=0.7, diou_threshold_3=0.7, conf_score_threshold=0.7):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.dij_theshold = dij_threshold
    self.diou_threshold_2 = diou_threshold_2
    self.diou_threshold_3 = diou_threshold_3
    self.conf_score_threshold = conf_score_threshold
    self.trackers = []
    self.frame_count = 0

  def update(self, dets=np.empty((0, 5))):
    """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    trks = np.zeros((len(self.trackers), 5))
    to_del = []
    ret = []
    for t, trk in enumerate(trks):
      pos = self.trackers[t].predict()[0]
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      if np.any(np.isnan(pos)):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)

    # divide high-score and low-score
    highscore_dets, lowscore_dets = divide_dets_byscore(dets, self.conf_score_threshold)

    # matching: following the pipeline from the paper
    # first stage
    matched1, unmatched_dets1, unmatched_trks = associate_detections_firststage(highscore_dets,trks,self.dij_theshold)
    # second stage
    matched2, unmatched_trks = associate_detections_secondstage(lowscore_dets,trks,unmatched_trks,self.diou_threshold_2)
    # third stage
    matched3, unmatched_dets3, unmatched_trks = associate_detections_thirdstage(highscore_dets,trks,unmatched_trks,unmatched_dets1,self.diou_threshold_3)


    # update matched trackers with assigned detections
    dets = np.concatenate((highscore_dets,lowscore_dets),axis=0)
    matched2[:,0]+=len(highscore_dets)
    matched = np.concatenate((matched1, matched3, matched2), axis=0)
    unmatched_dets = unmatched_dets3

    for m in matched:
      self.trackers[m[1]].update(dets[m[0], :])
    # create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
        trk = KalmanBoxTracker(dets[i,:])
        self.trackers.append(trk)
    i = len(self.trackers)
    for trk in reversed(self.trackers):
      d = trk.get_state()[0]
      if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
        ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
      i -= 1
      # remove dead tracklet
      if(trk.time_since_update > self.max_age):
        self.trackers.pop(i)
    i = len(self.trackers)

    if(len(ret)>0):
      return np.concatenate(ret)

    return np.empty((0,5))

