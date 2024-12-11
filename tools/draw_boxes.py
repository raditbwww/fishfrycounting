import cv2

# Draw Final Bbox
def draw_final_bbox_and_linecount(track_bbs_ids, fish_count, line_x_pos):
  for track in track_bbs_ids:
    x_min, y_min, x_max, y_max, track_id = track.astype(int)
    # Generate a unique color for each track based on its ID
    # color = generate_color(track_id)
    color = (255, 0, 0)
    thickness = 2
    # Draw the bounding box
    cv2.rectangle(im0, (x_min, y_min), (x_max, y_max), color, thickness)
    # Display the track ID
    cv2.putText(im0, f'ID: {track_id}', (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
  # Display the fish count
  cv2.putText(im0, f'fish_count: {fish_count}', (700,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)
  # Draw the vertical line on the frame
  cv2.line(im0, (line_x_pos, 0), (line_x_pos, h), (255, 0, 0), 5)  # Blue color (0, 255, 0) and 5 px thickness