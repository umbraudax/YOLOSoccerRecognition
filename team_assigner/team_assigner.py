import numpy as np
from sklearn.cluster import KMeans


class TeamAssigner:

    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {} # id: team_num

    def _new_kmeans_model(self):
        return KMeans(n_clusters=2, init="k-means++", n_init=1)


    def _preprocess_frame(self, frame, bbox):
        img = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        img_top_half = img[0:img.shape[0]//2, :]
        img_2d = img_top_half.reshape(-1, 3)
        return (img_top_half, img_2d)

    def get_player_color(self, frame, bbox):
        img_top_half, img_2d = self._preprocess_frame(frame, bbox)
        k_model = self._new_kmeans_model()
        k_model.fit(img_2d)

        clust_img = k_model.labels_.reshape(img_top_half.shape[0], img_top_half.shape[1])
        corner_clusters = [clust_img[0, 0], 
                   clust_img[-1, 0], 
                   clust_img[0, -1], 
                   clust_img[-1, -1]]
        player_cluster = 1 - max(set(corner_clusters), key=corner_clusters.count)
        return k_model.cluster_centers_[player_cluster]

    def assign_team_color(self, frame, player_detections):
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection['bbox']
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        team_model = self._new_kmeans_model()
        team_model.fit(np.array(player_colors))

        self.team_kmeans = team_model

        self.team_colors[1] = team_model.cluster_centers_[0]
        self.team_colors[2] = team_model.cluster_centers_[1]


    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox)
        team_id = self.team_kmeans.predict(player_color.reshape(1,-1))[0] + 1

        self.player_team_dict[player_id] = team_id
        return team_id


