import os

video_folder = os.path.join(os.getcwd(), 'demo', 'resources')
video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.MOV', '.mov'))]

for video_file in video_files:
    video_path = os.path.join(video_folder, video_file)

    command = f'python demo/topdown_demo_with_mmdet.py ' \
              f'demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py ' \
              f'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/' \
              f'rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth ' \
              f'configs/body_2d_keypoint/rtmpose/body8/' \
              f'rtmpose-m_8xb256-420e_body8-256x192.py ' \
              f'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/' \
              f'rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth ' \
              f'--input {video_path} ' \
              f'--output-root=vis_results/video_2d_poses ' \
              f'--save-predictions ' #\
            #   f'--show ' \
            #   f'--draw-heatmap '
    
    print(f'Start processing video {video_file}...')
    os.system(command)
    print(f' {video_file}')
