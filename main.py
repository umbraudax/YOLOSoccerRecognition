from utils import read_video, save_video
from tracker_helpers import Tracker
import cv2


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
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    # save_cropped_image(tracks, video_frames[0])

    # save_video(output_video_frames, 'output_videos/output_video.avi')
    # tracker.model.predict('input/08fd33_4.mp4', save=True, device='mps')




if __name__ == "__main__":
    main()