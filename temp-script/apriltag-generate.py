import cv2

aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)

tag_id = 0
pixels = 600  # 输出图片边长，像素越大打印越清晰
img = aruco.generateImageMarker(dictionary, tag_id, pixels)

cv2.imwrite("tag36h11_id0.png", img)
