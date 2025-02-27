import numpy as np
import pandas as pd

# 関節数
n_joints = 10

# フレーム数
n_frames = 100

# カメラ数 (ここでは3とする)
n_cameras = 1

# 各カメラの関節点データ (3次元)
camera_data = []
for _ in range(n_cameras):
    joint_data = np.zeros((n_joints, n_frames, 3))  # 3次元 (x, y, z)
    for j in range(n_joints):
        for f in range(n_frames):
            joint_data[j, f, 0] = f + j*2  # x座標: フレームごとに1増加、関節ごとにオフセット
            joint_data[j, f, 1] = f + j*3 # y座標: フレームごとに1増加、関節ごとにオフセット
            # z座標は0で固定 (簡単のため)
    camera_data.append(joint_data)

# pointListとして与えられる形(可視化コードの入力)
pointList = []

for i in range(len(camera_data)):
  point_list_per_cam = []
  for j in range(n_joints):
    joint_coords = camera_data[i][j]
    point_list_per_cam.append(joint_coords)

  pointList.append(np.array(point_list_per_cam))


# 最初のカメラの最初の関節点のデータを確認
print(pointList[0][0])

# 形状を確認 (関節点数, フレーム数, 3)
print(pointList[0].shape)

# triangulateMultiview関数から得られると想定されるデータ(簡単のため、ここではcamera_data[0]をそのまま使用)
points3d = camera_data[0]
print(points3d.shape)





# utilsCheker.pyのsynchronizeVideos参考
import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'

for i, data in enumerate(pointList):
    # nPoints, nFrames, _ = data.shape  # 不要な行
    # 3次元データを2次元データ(x, y)に変換
    data_2d = data[:, :, :2]  # z座標を削除
    nPoints, nFrames, _ = data_2d.shape

    # Reshape the 2D numpy array, preserving point and frame indices
    data_reshaped = data_2d.reshape(-1, 2)

    # Create DataFrame
    df = pd.DataFrame(data_reshaped, columns=['x', 'y'])

    # Add columns for point number and frame number
    df['Point'] = np.repeat(np.arange(nPoints), nFrames)
    df['Frame'] = np.tile(np.arange(nFrames), nPoints)

    # Reorder columns if needed
    df = df[['Point', 'Frame', 'x', 'y']]

    # Create a figure and add an animated scatter plot
    fig = px.scatter(
        df,
        x='x',
        y='y',
        title="Cam " + str(i),
        animation_frame='Frame',
        range_x=[0, 1200],
        range_y=[1200, 0],
        color='Point',
        color_continuous_scale=px.colors.sequential.Viridis,
    )

    # アニメーションの速度を変更 (2倍速)
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=[
                    dict(
                        args=[
                            None,
                            {
                                "frame": {"duration": 1000 / 120, "redraw": True},  # durationを半分にする
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                        label="Play",
                        method="animate",
                    )
                ],
                direction="left",
                pad={"r": 10, "t": 87},
                showactive=False,
                type="buttons",
                x=0.1,
                xanchor="right",
                y=0,
                yanchor="top",
            )
        ]
    )

    # Show the animation
    fig.show()