from utils import read_video, save_video
from tracker_helpers import Tracker
import cv2
from team_assigner import TeamAssigner
from player_assigner import PlayerAssigner
import numpy as np


VIDEO_NAME = "video1"

def save_cropped_image(tracks, frame):
    # save cropped player image
    for track_id, player in tracks['players'][0].items():
        
        if not player: continue
        bbox = player['bbox']
        cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        cv2.imwrite(f'output_videos/playerimage_{track_id}.jpg', cropped_image)


        break


def main():
    video_frames = read_video(f'input/{VIDEO_NAME}.mp4')

    tracker = Tracker('./models/best.pt')

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=False,
                                       stub_path=f'stubs/{VIDEO_NAME}.pkl')
    # save_cropped_image(tracks, video_frames[0])

    # interpolate ball pos
    tracks['ball'] = tracker.interpolate_ball_pos(tracks['ball'])

    INIT_FRAME = 3
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[INIT_FRAME],
                                    tracks['players'][INIT_FRAME])

    player_assigner = PlayerAssigner()
    
    
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        closest_player = player_assigner.assign_player(tracks['players'][frame_num], tracks['ball'][frame_num][0])
        for player_id, track in player_track.items():
        
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track['bbox'],
                                                 player_id)
            track['team'] = team
            track['team_color'] = team_assigner.team_colors[team]
            track['has_ball'] = closest_player and closest_player != -1 and player_id == closest_player
            if track['has_ball']:
                team_ball_control.append(track['team'])
        if not team_ball_control:
             team_ball_control.append(1)
        elif not closest_player or closest_player == -1:
                team_ball_control.append(team_ball_control[-1])
    team_ball_control = np.array(team_ball_control)

        
        
            

    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)



    save_video(output_video_frames, 'output_videos/output_video.avi')
    # tracker.model.predict('input/08fd33_4.mp4', save=True, device='mps')




if __name__ == "__main__":
    main()