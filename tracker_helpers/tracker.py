from ultralytics import YOLO
import supervision as sv
from trackers import ByteTrackTracker
import pickle
import os
import sys
import cv2
import numpy as np
import pandas as pd

sys.path.append('../')
from utils import get_center_of_bbox, get_width_of_bbox, get_foot_position

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = ByteTrackTracker()


    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.1, device='mps')
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks



        detections = self.detect_frames(frames)
        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)

            # convert goalkeeper classifications to players for consistency
            for object_idx, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_idx] = cls_names_inv["player"]

            # track objects
            detections_with_tracks = self.tracker.update(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detections_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][0] = {"bbox": bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id):
        y2 = int(bbox[3])

        x_center, _ = get_center_of_bbox(bbox)
        width = get_width_of_bbox(bbox)

        cr, cg, cb = color
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=(cb,cg,cr),
            thickness=2,
            lineType=cv2.LINE_4
        )
        rect_width = 40
        rect_height = 20
        x1_rect = x_center - rect_width // 2
        y1_rect = y2 - rect_height // 2
        x2_rect = x_center + rect_width // 2
        y2_rect = y2 + rect_height // 2

        if track_id is not None:
            cv2.rectangle(frame,
                          (x1_rect, y1_rect),
                          (x2_rect, y2_rect),
                          (cb,cg,cr),
                          cv2.FILLED)

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255,255,255) if color[0] == 0 and color[1] == 0 and color[2] == 0 else (0, 0, 0),
                2
            )

        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x, y],
            [x-10, y-20],
            [x+10, y-20]
        ], np.int32)

        cr, cg, cb = color
        cv2.drawContours(frame,
                      [triangle_points],
                      0,
                      (cb, cg, cr),
                      cv2.FILLED)
        cv2.drawContours(frame,
                        [triangle_points],
                        0,
                        (0,0,0),
                        2)

        return frame


    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # draw semi-transparent rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), cv2.FILLED)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        # get number of times each team had ball
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]
        cv2.putText(
                    frame,
                    f"Team 1 Ball Control {team_1_num_frames/(frame_num+1):.2f}",
                    (1400, 900),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,0,0),
                    2
                )
        cv2.putText(
                    frame,
                    f"Team 2 Ball Control {team_2_num_frames/(frame_num+1):.2f}",
                    (1400, 950),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,0,0),
                    2
                )
        return frame
        


    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks['players'][frame_num]
            ball_dict = tracks['ball'][frame_num]
            referee_dict = tracks['referees'][frame_num]

            for track_id, player in player_dict.items():
                frame = self.draw_ellipse(frame, player['bbox'], player['team_color'], track_id)
                if player['has_ball']:
                    self.draw_triangle(frame, player['bbox'], (255, 0, 0))

            for track_id, referee in referee_dict.items():
                frame = self.draw_ellipse(frame, referee['bbox'], (0, 0, 0), track_id)

            if ball_dict:
                frame = self.draw_triangle(frame, ball_dict[0]['bbox'], (0, 0, 255))


            # draw team ball control
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)
            
            output_video_frames.append(frame)
        return output_video_frames

    def interpolate_ball_pos(self, ball_positions):
        ball_positions = [x.get(0, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{0: {'bbox': x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def add_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    track_info['position'] = position


            