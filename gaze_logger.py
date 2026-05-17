# gaze_logger.py
import csv

def init_csv(csv_file):
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Frame', 'Gaze_X', 'Gaze_Y', 'Fixation', 'Side'])

def append_gaze_data(csv_file, frame_count, gaze_vec, fixation, side):
    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([frame_count,
                         round(gaze_vec[0], 4),
                         round(gaze_vec[1], 4),
                         fixation,
                         side])