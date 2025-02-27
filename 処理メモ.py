import numpy as np
import cv2
import os
import glob
import copy
import pickle

def saveCameraParameters(filename,CameraParams):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename),exist_ok=True)

    open_file = open(filename, "wb")
    pickle.dump(CameraParams, open_file)
    open_file.close()

    return True

def generate3Dgrid(CheckerBoardParams):
    #  3D points real world coordinates. Assuming z=0
    objectp3d = np.zeros((1, CheckerBoardParams['dimensions'][0]
                          * CheckerBoardParams['dimensions'][1],
                          3), np.float32)
    objectp3d[0, :, :2] = np.mgrid[0:CheckerBoardParams['dimensions'][0],
                                    0:CheckerBoardParams['dimensions'][1]].T.reshape(-1, 2)

    objectp3d = objectp3d * CheckerBoardParams['squareSize']

    return objectp3d

def calcIntrinsics(folderName, CheckerBoardParams=None, filenames=['*.jpg'],
                   imageScaleFactor=1, visualize=False, saveFileName=None):
    if CheckerBoardParams is None:
        # number of black to black corners and side length (cm)
        CheckerBoardParams = {'dimensions': (6,9), 'squareSize': 2.71}

    if '*' in filenames[0]:
        imageFiles = glob.glob(folderName + '/' + filenames[0])

    else:
        imageFiles = [] ;
        for fName in filenames:
            imageFiles.append(folderName + '/' + fName)

    # stop the iteration when specified
    # accuracy, epsilon, is reached or
    # specified number of iterations are completed.
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Vector for 3D points
    threedpoints = []

    # Vector for 2D points
    twodpoints = []

    #  3D points real world coordinates
    # objectp3d = generate3Dgrid(CheckerBoardParams)

    # Load images in for calibration
    for iImage, pathName in enumerate(imageFiles):
        image = cv2.imread(pathName)
        if imageScaleFactor != 1:
            dim = (int(imageScaleFactor*image.shape[1]),int(imageScaleFactor*image.shape[0]))
            image = cv2.resize(image,dim,interpolation=cv2.INTER_AREA)
        imageSize = np.reshape(np.asarray(np.shape(image)[0:2]).astype(np.float64),(2,1)) # This all to be able to copy camera param dictionary

        grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print(pathName + ' used for intrinsics calibration.')

        # Find the chess board corners
        # If desired number of corners are
        # found in the image then ret = true
        # ret, corners = cv2.findChessboardCorners(
        #                 grayColor, CheckerBoardParams['dimensions'],
        #                 cv2.CALIB_CB_ADAPTIVE_THRESH
        #                 + cv2.CALIB_CB_FAST_CHECK +
        #                 cv2.CALIB_CB_NORMALIZE_IMAGE)

        ret,corners,meta = cv2.findChessboardCornersSBWithMeta(	grayColor, CheckerBoardParams['dimensions'],
                                                        cv2.CALIB_CB_EXHAUSTIVE +
                                                        cv2.CALIB_CB_ACCURACY +
                                                        cv2.CALIB_CB_LARGER)

        # If desired number of corners can be detected then,
        # refine the pixel coordinates and display
        # them on the images of checker board
        if ret == True:
            # 3D points real world coordinates
            checkerCopy = copy.copy(CheckerBoardParams)
            checkerCopy['dimensions'] = meta.shape[::-1] # reverses order so width is first
            objectp3d = generate3Dgrid(checkerCopy)

            threedpoints.append(objectp3d)

            # Refining pixel coordinates
            # for given 2d points.
            # corners2 = cv2.cornerSubPix(
            #     grayColor, corners, (11, 11), (-1, -1), criteria)

            corners2 = corners/imageScaleFactor # Don't need subpixel refinement with findChessboardCornersSBWithMeta
            twodpoints.append(corners2)

            # Draw and display the corners
            image = cv2.drawChessboardCorners(image,
                                                meta.shape[::-1],
                                                corners2, ret)

            #findAspectRatio
            ar = imageSize[1]/imageSize[0]
            # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
            cv2.resize(image,(int(600*ar),600))

            # Save intrinsic images
            imageSaveDir = os.path.join(folderName,'IntrinsicCheckerboards')
            if not os.path.exists(imageSaveDir):
                os.mkdir(imageSaveDir)
            cv2.imwrite(os.path.join(imageSaveDir,'intrinsicCheckerboard' + str(iImage) + '.jpg'), image)

            if visualize:
                print('Press enter or close image to continue')
                cv2.imshow('img', image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        if ret == False:
            print("Couldn't find checkerboard in " + pathName)

    if len(twodpoints) < .5*len(imageFiles):
       print('Checkerboard not detected in at least half of intrinsic images. Re-record video.')
       return None


    # Perform camera calibration by
    # passing the value of above found out 3D points (threedpoints)
    # and its corresponding pixel coordinates of the
    # detected corners (twodpoints)
    ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(
        threedpoints, twodpoints, grayColor.shape[::-1], None, None)

    CamParams = {'distortion':distortion,'intrinsicMat':matrix,'imageSize':imageSize}

    if saveFileName is not None:
        saveCameraParameters(saveFileName,CamParams)

    return CamParams






def nview_linear_triangulations(cameras, image_points,weights=None):
    """
    Computes world coordinates from image correspondences in n views.
    :param cameras: pinhole models of cameras corresponding to views
    :type cameras: sequence of Camera objects
    :param image_points: image coordinates of m correspondences in n views
    :type image_points: sequence of m numpy.ndarray, shape=(2, n)
    :return: m world coordinates
    :rtype: numpy.ndarray, shape=(3, m)
    :weights: numpy.ndarray, shape(nMkrs,nCams)
    """
    assert(type(cameras) == list)
    assert(type(image_points) == list)
    assert(len(cameras) == image_points[0].shape[1])
    assert(image_points[0].shape[0] == 2)

    world = np.zeros((3, len(image_points)))
    confidence = np.zeros((1,len(image_points)))
    for i, correspondence in enumerate(image_points):
        if weights is not None:
            w = [w[i] for w in weights]
        else:
            w = None
        pt3d, conf = nview_linear_triangulation(cameras, correspondence,weights=w)
        world[:, i] = np.ndarray.flatten(pt3d)
        confidence[0,i] = conf
    return world,confidence

def nview_linear_triangulation(cameras, correspondences,weights = None):
    """
    Computes ONE world coordinate from image correspondences in n views.
    :param cameras: pinhole models of cameras corresponding to views
    :type cameras: sequence of Camera objects
    :param correspondences: image coordinates correspondences in n views
    :type correspondences: numpy.ndarray, shape=(2, n)
    :return: world coordinate
    :rtype: numpy.ndarray, shape=(3, 1)
    """
    assert(len(cameras) >= 2)
    assert(type(cameras) == list)
    assert(correspondences.shape == (2, len(cameras)))

    def _construct_D_block(P, uv,w=1):
        """
        Constructs 2 rows block of matrix D.
        See [1, p. 88, The Triangulation Problem]
        :param P: camera matrix
        :type P: numpy.ndarray, shape=(3, 4)
        :param uv: image point coordinates (xy)
        :type uv: numpy.ndarray, shape=(2,)
        :return: block of matrix D
        :rtype: numpy.ndarray, shape=(2, 4)
        """

        return w*np.vstack((uv[0] * P[2, :] - P[0, :],
                          uv[1] * P[2, :] - P[1, :]))

    # testing weighted least squares(カメラごとの重みを設定している場合はその重みを使用)
    if weights is None:
        w = np.ones(len(cameras))
        weights = [1 for i in range(len(cameras))]
    else:
        w = [np.nan_to_num(wi,nan=0.5) for wi in weights] # turns nan confidences into 0.5

    # D をゼロ行列 (形状: (カメラの数 * 2, 4)) として初期化します。
    # カメラのリスト、対応点の座標、カメラのインデックスを zip でまとめてループします。
    # 各カメラについて、_construct_D_block 関数を呼び出し、D の対応する2行を計算します。
    # cam.P は、Camera オブジェクト cam の投影行列 (3x4) を取得します。
    # uvはそのカメラにおける対応点のxy座標
    # *　w[cam_idx]はそのカメラの信頼度
    # 計算されたブロックを D に格納します。
    D = np.zeros((len(cameras) * 2, 4))
    for cam_idx, cam, uv in zip(range(len(cameras)), cameras, correspondences.T):
        D[cam_idx * 2:cam_idx * 2 + 2, :] = _construct_D_block(cam.P, uv,w=w[cam_idx])
    # 特異値分解 (SVD):
    # D の転置と D の積である Q を計算します (Q = D.T.dot(D))。
    # Q に対して特異値分解 (SVD) を実行します (u, s, vh = np.linalg.svd(Q))。
    # u の最後の列を p2e 関数（同次座標をユークリッド座標に変換する関数、別途定義が必要）に渡し、3次元点 pt3d を計算します。
    Q = D.T.dot(D)
    u, s, vh = np.linalg.svd(Q)
    pt3d = p2e(u[:, -1, np.newaxis])
    # 信頼度計算:
    # 2つ以上のカメラが0でない信頼度を持っているかをチェックし、もし持っていない場合は3次元座標を0ベクトル、信頼度を0として返します。
    # 全ての信頼度がNaNである(=全てのカメラが補間されたものである)場合、信頼度confを0.5とします。
    # それ以外の場合、0でない信頼度の平均値を計算しconfとします。
    weightArray = np.asarray(weights)
    if np.count_nonzero(weights)<2:
        # return 0s if there aren't at least 2 cameras with confidence
        pt3d = np.zeros_like(pt3d)
        conf = 0
    else:
        # if all nan slice (all cameras were splined)
        if all(np.isnan(weightArray[weightArray!=0])):
            conf=.5 # nans get 0.5 confidence
        else:
            conf = np.nanmean(weightArray[weightArray!=0])

    return pt3d,conf

def p2e(projective):  #同次座標 (4要素) をユークリッド座標 (3要素) に変換
    """
    Convert 2d or 3d projective to euclidean coordinates.
    :param projective: projective coordinate(s)
    :type projective: numpy.ndarray, shape=(3 or 4, n)
    :return: euclidean coordinate(s)
    :rtype: numpy.ndarray, shape=(2 or 3, n)
    """
    assert(type(projective) == np.ndarray)
    assert((projective.shape[0] == 4) | (projective.shape[0] == 3))
    return (projective / projective[-1, :])[0:-1, :]