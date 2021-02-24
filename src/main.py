import cv2

from object_eraser import ObjectEraser

if __name__ == '__main__':
    eraser = ObjectEraser(size=(680,512))
    result, mask = eraser.erase("assets/img.png", "assets/mask.png")
    cv2.imwrite("results/result.png", result)