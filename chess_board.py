import cv2 as cv
import numpy as np

# 비디오 입력 받기
cap = cv.VideoCapture('chess.mp4')

# 체스보드 이미지 크기
board_pattern = (9, 6)
board_cellsize = 2.5  # cm

# 선택된 이미지를 저장할 리스트
img_select = []

# 카메라 보정 함수
def calib_camera_from_chessboard(images, board_pattern, board_cellsize, K=None, dist_coeff=None, calib_flags=None):
    # 체스보드 코너점 3D 좌표 생성
    objp = np.zeros((board_pattern[0] * board_pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_pattern[0], 0:board_pattern[1]].T.reshape(-1, 2) * board_cellsize
    
    # 2D 코너점 검출
    obj_points = []
    img_points = []
    for img in images:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        complete, corners = cv.findChessboardCorners(gray, board_pattern)
        if complete:
            obj_points.append(objp)
            img_points.append(corners)

    # 카메라 보정
    rms, K, dist_coeff, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, gray.shape[::-1], K, dist_coeff, flags=calib_flags)

    return rms, K, dist_coeff, rvecs, tvecs


while True:
    # 비디오 프레임 읽기
    ret, frame = cap.read()

    # 체스보드 코너 검출
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    complete, corners = cv.findChessboardCorners(gray, board_pattern)

    if complete:
        # 체스보드 코너점 그리기
        cv.drawChessboardCorners(frame, board_pattern, corners, complete)

        # 이미지 선택하기
        if len(img_select) < 10:
            img_select.append(frame.copy())
            cv.putText(frame, f'Select: {len(img_select)}', (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))
        else:
            # 카메라 보정
            rms, K, dist_coeff, rvecs, tvecs = calib_camera_from_chessboard(img_select, board_pattern, board_cellsize)

            # 오차 출력
            print(f'RMS: {rms}')
            print(f'K:\n{K}')
            print(f'Distortion coefficients: {dist_coeff.ravel()}')

            # PnP 풀기
            objp = np.random.randint(0, 10, (board_pattern[0] * board_pattern[1], 3)).astype(np.float32)
            _, rvecs, tvecs = cv.solvePnP(objp, corners, K, dist_coeff)

            # 결과 출력
            print(f'Rotation Vector:\n{rvecs}')
            print(f'Translation Vector:\n{tvecs}')

            # 이미지 선택 초기화
            img_select = []

    # 화면 출력
    cv.imshow('frame', frame)

    # 종료
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
