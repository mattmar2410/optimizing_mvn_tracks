#!/usr/bin/env python
"""
A ROS node to track objects via TensorFlow Object Detection API.

Author:
    Tenzing thjoshi -- thjoshi@lbl.gov
"""
import numpy as np
from sort import sort_optimize_trks

class ObjectTrackerNode(object):
    """docstring for PeopleObjectDetectionNode."""
    def __init__(self, R, qp, qv, cov_v):

        # get the parameters
        (detection_topic, tracker_topic, match_threshold, filter_threshold,
         distance_metric,
         max_age, min_hits, labels, tracker_label,
         track_frame_id, output_frame_id,
         R, qp, qv, cov_v,
         tracker_type, qtplot) = self.get_parameters(R, qp, qv, cov_v)

        self.tracker = []
        for ll in labels:
            self.tracker.append(sort_optimize_trks.Sort(max_age=max_age,
                                match_threshold=match_threshold,
                                filter_threshold=filter_threshold,
                                R=R,
                                qp=qp,
                                qv=qv,
                                cov_v=cov_v,
                                tracker_type=tracker_type,
                                qtplot=qtplot))
        self.labels = labels
        self.min_hits = min_hits
        self.tracker_label = tracker_label
        self.track_frame_id = track_frame_id
        self.output_frame_id = output_frame_id
        self.last_time = 0


        # create color codes for tracks
        np.random.seed(11)
        self.track_colors = np.random.random(size=(100, 3))


    def get_parameters(self, R,
                       qp, qv, cov_v):
        """
        Gets the necessary parameters from parameter server
        Args:
        Returns:
        (tuple) (detection_topic, tracker_topic, match_threhold,
                 filter_threshold, max_age, min_hits, labels, tracker_label
                 spatial_tracking)
        """

        print('optimizing for vehicles')
        detection_topic = "/Detections"
        tracker_topic = "/object_tracker/tracks"
        match_threhold = 0.8
        filter_threshold = 0.2
        distance_metric = 'Hellinger'
        min_hits = 1
        max_age = 3
        labels = ["ped/cyc", "vehicle"]
        tracker_label = ['person', 'vehicle']
        # Optimizing with LiDAR tracks and LiDAR is already in base_link
        track_frame_id = None
        output_frame_id = None
        R=R
        qp=qp
        qv=qv
        cov_v=cov_v
        tracker_type = "mvn"
        qtplot = False
        assert len(labels) == len(tracker_label), (
            "'labels' and 'tracker_labels' must be of same length")

        return (detection_topic, tracker_topic, match_threhold,
                filter_threshold, distance_metric, 
                max_age, min_hits, labels, tracker_label,
                track_frame_id, output_frame_id, R, qp, qv, cov_v,
                tracker_type, qtplot)


    def detection_callback(self, detections):
        """
        Callback for RGB images
        Args:
        detections: detections array message from cob package
        image: rgb frame in openCV format
        """
        cur_time = 0
        for hdr in detections.image_headers:
            cur_time += hdr.stamp.to_sec()
        if cur_time:
            cur_time /= len(detections.image_headers)

        dt = cur_time - self.last_time
        self.last_time = cur_time

        # initiate tracks array and matches array
        found_tracks, labels, types, matches = [], [], [], []

        # loop over all possible labels
        for lidx, lnames in enumerate(self.labels):

            # extract all detections with that label
            det_list = []
            det_map = []

            for didx, detection in enumerate(detections.detections):

                # here we abuse that both "label" in "label" and
                # "label" in ["label"] are evaluated to 'true'
                if detection.label in lnames:
                    # only want to optimize for vehicles
                    if detection.label == "ped/cyc":
                        continue
                    # vel var only used for initialization at this point
                    covariance = np.zeros((6, 6))

                    center = [detection.bbox.pose.pose.position.x,
                              detection.bbox.pose.pose.position.y,
                              detection.bbox.pose.pose.position.z]
                    size = [detection.bbox.size.x * 0.5,
                            detection.bbox.size.y * 0.5,
                            detection.bbox.size.z * 0.5]
                    cov = np.asarray(detection.bbox.pose.covariance)
                    cov = cov.reshape((6, 6))
                    covariance[:3, :3] = cov[:3, :3]
                    x = center[0] - size[0]
                    y = center[1] - size[1]
                    z = center[2] - size[2]
                    x2 = center[0] + size[0]
                    y2 = center[1] + size[1]
                    z2 = center[2] + size[2]
                    score = detection.score
                    three_dim = detection.bbox.three_dimensional

                    # Hold onto the size for mvn purposes
                    data = {'position': np.array([x, y, z,
                                                  x2, y2, z2]),
                            'covariance': np.array(covariance),
                            'score': np.array(score),
                            'size': size,
                            'label': detection.label}

                    if self.tracker[lidx].type == "3d" and three_dim:
                        pass
                    elif self.tracker[lidx].type == "mvn" and three_dim:
                        data['position'] = np.array([center[0], center[1],
                                                     center[2]])
                    elif self.tracker[lidx].type == "2d" and not three_dim:
                        data['position'] = np.array([x, y, x2, y2])
                    else:
                        rospy.logerr("Enter proper spatial_tracking label"
                                     "2D, 3D, mvn")

                    trans = None
                    if self.track_frame_id is not None:
                        # TODO: is this good if multiple headers?
                        #       is it possible?
                        try:
                            '''
                            wiki.ros.org/tf/Tutorials/Writing%20a%20tf%20listener%20%28Python%29
                            lookup_transform will give the rotation from
                            the parent orientation
                            (camera) to the child orientation (base_link)
                            lookup_transform(child_frame, parent_frame, time)
                            '''
                            trans = self.tfBuffer.lookup_transform(
                                self.track_frame_id, 
                                detection.header.frame_id,
                                detection.header.stamp)
                            #trans = self.tfBuffer.lookup_transform(
                            #        "os1_lidar_link",
                            #    self.track_frame_id, 
                            #    detection.header.stamp)
                            #print('transform backwards')
                        except (tf2_ros.LookupException,
                                tf2_ros.ConnectivityException,
                                tf2_ros.ExtrapolationException):

                            # if we ask for a transform but don't get one
                            # we will just go to the next detection
                            # and don't include this one, is this smart?
                            continue
                        if trans is not None:
                            data = tf_transform(data, trans)

                    det_list.append(data)
                    det_map.append(didx)


            # do the actual sort update
            tracks = self.tracker[lidx].update(np.array(det_list), dt=dt)

            # extract found_tracks -> detection mapping
            trk_map = [[] for _ in range(len(tracks))]
            for tidx, didx in zip(self.tracker[lidx].matches, det_map):
                trk_map[tidx].append(didx)

            # add data to the arrays used later for creating the track message
            found_tracks.extend(tracks)
            matches.extend(trk_map)
            labels.extend([self.tracker_label[lidx]]*len(tracks))
            types.extend([self.tracker[lidx].type]*len(tracks))

        return found_tracks
