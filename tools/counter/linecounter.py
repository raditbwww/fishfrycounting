# Counting
# as long as bboxes still touch the line and ID not counted yet, then increment counter

# input: list of array of tracks with trk of tracks=[x1,y1,x2,y2,id]
# output: count of fish

def count_fish(line_x_pos, track_bbs_ids, ID_already_counted, fish_count):
  for track in track_bbs_ids:
    ID = int(track[4])
    right_x = int(track[2])
    left_x = int(track[0])
    if (ID not in ID_already_counted) and (right_x>line_x_pos) and (left_x<line_x_pos):
        fish_count+=1
        ID_already_counted.append(ID)
  return fish_count