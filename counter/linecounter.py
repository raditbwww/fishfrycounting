# count by bboxes width
# as long as bboxes still touch the line and ID not counted yet, then increment counter

# input: list of array of tracks with trk of tracks=[x1,y1,x2,y2,id]
# output: count of fish

def counter(track_bbs_ids, print_track_bbs=True):
    for track in track_bbs_ids:
        if (print_track_bbs==True):
            x_min, y_min, x_max, y_max, track_id = track.astype(int)
            # Generate a unique color for each track based on its ID
            # color = generate_color(track_id)
            color = (255, 0, 0)
            thickness = 2
            # Draw the bounding box
            cv2.rectangle(im0, (x_min, y_min), (x_max, y_max), color, thickness)
            # Display the track ID
            cv2.putText(im0, f'ID: {track_id}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2) 

        ID = int(track[4])
        right_x = int(track[2])
        left_x = int(track[0])

        if (ID not in ID_already_counted) and (right_x>line_x_pos) and (left_x<line_x_pos):
            fish_count+=1
            ID_already_counted.append(ID)