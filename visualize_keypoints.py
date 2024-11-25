import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os


script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


def visualize_3d_frame(json_file, frame_idx=0):
    # Read JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Get metadata
    meta_info = data['meta_info']
    skeleton_links = meta_info['skeleton_links']
    keypoint_colors = np.array(meta_info['keypoint_colors']['__ndarray__'])
    skeleton_colors = np.array(meta_info['skeleton_link_colors']['__ndarray__'])
    
    # Get instances for specified frame
    frame_data = data['instance_info'][frame_idx]['instances']
    
    # Create 3D figure
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw keypoints for each detected person
    for person in frame_data:
        keypoints = np.array(person['keypoints'])
        scores = np.array(person['keypoint_scores'])
        
        # Extract x, y, z coordinates
        x = keypoints[:, 0]
        y = keypoints[:, 1]
        z = keypoints[:, 2]
        
        # Draw skeleton connections
        for link_idx, (start_idx, end_idx) in enumerate(skeleton_links):
            if scores[start_idx] > 0.3 and scores[end_idx] > 0.3:
                color = skeleton_colors[link_idx] / 255.0
                ax.plot([x[start_idx], x[end_idx]],
                       [y[start_idx], y[end_idx]],
                       [z[start_idx], z[end_idx]],
                       color=color, linewidth=2)
        
        # Draw keypoints
        for i in range(len(keypoints)):
            if scores[i] > 0.3:
                color = keypoint_colors[i] / 255.0
                ax.scatter(x[i], y[i], z[i], c=[color], s=50)
    
    # Set figure properties
    ax.set_title(f'Frame {frame_idx + 1}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Adjust view angle
    ax.view_init(elev=10, azim=45)
    
    # Set coordinate axis ranges
    ax.set_xlim([-0.8, 0.8])
    ax.set_ylim([-0.8, 0.8])
    ax.set_zlim([0, 2])
    
    # Keep coordinate axis scale consistent
    ax.set_box_aspect([1,1,1])
    
    plt.show()


def visualize_single_pose(json_file, frame_idx=0, person_idx=0):
    # Read data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Get metadata and skeleton information
    meta_info = data['meta_info']
    skeleton_links = meta_info['skeleton_links']
    keypoint_colors = np.array(meta_info['keypoint_colors']['__ndarray__']) / 255.0
    skeleton_colors = np.array(meta_info['skeleton_link_colors']['__ndarray__']) / 255.0
    
    # Get pose data for single person
    pose_data = data['instance_info'][frame_idx]['instances'][person_idx]
    keypoints = np.array(pose_data['keypoints'])
    scores = np.array(pose_data['keypoint_scores'])
    
    # Create 3D figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw skeleton connections
    for link_idx, (start_idx, end_idx) in enumerate(skeleton_links):
        if scores[start_idx] > 0.3 and scores[end_idx] > 0.3:
            ax.plot([keypoints[start_idx,0], keypoints[end_idx,0]],
                   [keypoints[start_idx,1], keypoints[end_idx,1]],
                   [keypoints[start_idx,2], keypoints[end_idx,2]],
                   color=skeleton_colors[link_idx], linewidth=2)
    
    # Draw keypoints
    for i, (x, y, z) in enumerate(keypoints):
        if scores[i] > 0.3:
            ax.scatter(x, y, z, c=[keypoint_colors[i]], s=50)
    
    # Set view
    ax.set_title(f'Frame {frame_idx + 1}, Person {person_idx + 1}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=10, azim=45)
    
    # Set coordinate axis ranges and scale
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([0, 1.6])
    ax.set_box_aspect([1,1,1.6])
    
    plt.show()


if __name__ == '__main__':

    visualize_3d_frame('vis_results/results_0.json', frame_idx=5)

    visualize_3d_frame('vis_results/results_VID_20241101_145409.json', frame_idx=50)

    visualize_3d_frame('vis_results/results_VID_20241101_145409_clipped.json', frame_idx=100)
