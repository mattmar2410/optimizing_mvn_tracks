import rospy
import numpy as np
import rosbag
from tracker_node_opt import ObjectTrackerNode
import matplotlib.pyplot as plt
from sort import kalman_optimize_mvn_tracker
'''
This code is designed to optimize the MVN KF tracking.
It uses a modified kalman_tracker.py to edit the function
The function is using detections from Second_ROS to optimize
the KF for vehicles
'''

# Initialize variables
bagfilename = '/media/nvidia/murslbl2/LEMURS/static_rfs_2/pp_detections_minute_24_to_24_2020-04-27-13-52-05.bag'

# Parameters to check
distance = 0.8
cov_v = np.linspace(0.0, 10.0, 20)
meas_noise = 0.15 # measurement noise does not have an effect
qv = np.linspace(0.00001, 0.5, 20)
qp = np.linspace(0.01, 0.50, 20)


def read_bag(bagfilename, meas_noise, qp, qv, cov_v):
    h = ObjectTrackerNode(meas_noise, qp, qv, cov_v)
    h.get_parameters(meas_noise, qp, qv, cov_v)
    prev_det = 0
    with rosbag.Bag(bagfilename, 'r') as bag:
        for topic, msg, ts in bag:
            #if topic != '/depth_extraction/detections':
            if topic != '/Detections':
                continue
            # Unpack message
            found_tracks = h.detection_callback(msg)
            for lidx in range(len(found_tracks)):
                track_id = int(found_tracks[lidx]['track_id'])
                print(track_id, prev_det)
                if track_id > prev_det:
                    tot_trk_id = track_id
                    prev_det = track_id
    return tot_trk_id

# Optimize Filter

trk_list = []
dist_list = []
meas_noise_list = []
qp_list = []
qv_list = []
cov_v_list = []

for p in qp:
    for v in qv:
        for unc_v in cov_v:
            trk = read_bag(bagfilename, meas_noise, p, v, unc_v)
            print(trk, meas_noise, p, v, unc_v)

            # Create the track list
            trk_list.append(trk)
            meas_noise_list.append(meas_noise)
            qp_list.append(p)
            qv_list.append(v)
            cov_v_list.append(unc_v)
            # Reset the counter for the next iteration
            kalman_optimize_mvn_tracker.KalmanTracker.count = 1

values = {'tracks': trk_list,
          'HD_dist': distance,
          'meas_noise': meas_noise_list,
          'qp': qp_list,
          'qv': qv_list,
          'cov_v': cov_v_list}

#optimize_params = {'distance': distance, 'tracks': trk_list}
np.save('/home/nvidia/catkin_ws/src/tfod_ros/scripts/opt_params_md',
        values)

#plt.figure()
#plt.plot(distance, trk_list)
#plt.show()


