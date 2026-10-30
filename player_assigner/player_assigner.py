import numpy as np
from utils import get_center_of_bbox, distance

class PlayerAssigner:
    def __init__(self):
        self.max_ball_distance = 70

    def assign_player(self, player_tracks, ball_track):
        ball_x, ball_y = get_center_of_bbox(ball_track['bbox'])

        min_distance = float('inf')
        closest_player = None

        for player_id, track in player_tracks.items():
            left_distance = distance(ball_x, ball_y, track['bbox'][0], track['bbox'][3])
            right_distance = distance(ball_x, ball_y, track['bbox'][2], track['bbox'][3])

            candidate_distance = min(left_distance, right_distance)
            if candidate_distance < self.max_ball_distance and candidate_distance < min_distance:
                closest_player = player_id
        return closest_player
            