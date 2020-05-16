import cv2
import numpy as np
GRID_SIZE = 100

def draw_face(image, iface, color):
    image[:,GRID_SIZE*3*iface:GRID_SIZE*3*(iface+1),:] = color

def draw_meshes(image):
    for i in range(4):
        y = i*GRID_SIZE
        cv2.line(image, (0,y-1), (image.shape[1],y-1), (0,0,0), 1)
        cv2.line(image, (0,y), (image.shape[1],y), (0,0,0), 1)
    for j in range(19):
        x = j*GRID_SIZE
        cv2.line(image, (x-1,0), (x-1,image.shape[0]), (0,0,0), 1)
        cv2.line(image, (x,0), (x,image.shape[0]), (0,0,0), 1)

def draw_texts(image):
    for nface in range(6):
        for i in range(3):
            for j in range(3):
                y = i*GRID_SIZE+GRID_SIZE//3*2
                x = (nface*3+j)*GRID_SIZE+GRID_SIZE//3
                cv2.putText(image, str(i*3+j+1), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,0), 2, cv2.LINE_AA)

def draw(image):
    draw_face(image, 0, (255,255,255))
    draw_face(image, 1, (255,255,0))
    draw_face(image, 2, (255,0,0))
    draw_face(image, 3, (255,128,0))
    draw_face(image, 4, (0,255,0))
    draw_face(image, 5, (0,0,255))
    draw_meshes(image)
    draw_texts(image)

image = np.zeros((GRID_SIZE*3,GRID_SIZE*18,3), dtype=np.uint8)
draw(image)
cv2.imwrite('texture.jpg', image[...,::-1])
