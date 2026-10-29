from utils import read_video, save_video
from tracker_helpers import Tracker
import cv2
from team_assigner import TeamAssigner


def save_cropped_image(tracks, frame):
    # save cropped player image
    for track_id, player in tracks['players'][0].items():
        
        if not player: continue
        bbox = player['bbox']
        cropped_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        cv2.imwrite(f'output_videos/playerimage_{track_id}.jpg', cropped_image)


        break


def main():
    video_frames = read_video('input/08fd33_4.mp4')

    tracker = Tracker('./runs/detect/complete_model/weights/best.pt')

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')
    # save_cropped_image(tracks, video_frames[0])


    INIT_FRAME = 10
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[INIT_FRAME],
                                    tracks['players'][INIT_FRAME])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track['bbox'],
                                                 player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

    output_video_frames = tracker.draw_annotations(video_frames, tracks)



    save_video(output_video_frames, 'output_videos/output_video.avi')
    # tracker.model.predict('input/08fd33_4.mp4', save=True, device='mps')




if __name__ == "__main__":
    main()