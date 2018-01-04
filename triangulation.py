import cv2
import numpy as np
import random
import sys

# Check if a point is inide a rectangle
def rect_contains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[2]:
        return False
    elif point[1] > rect[3]:
        return False
    return True

# Draw a point
def draw_point(img, p, color):
    cv2.circle(img, p, 2, color, cv2.FILLED, cv2.LINE_AA, 0)

# Draw delaunay triangles
def draw_delaunay(img, subdiv, delaunay_color, points):
    triangle_indices = []
    triangleList = subdiv.getTriangleList();

    # print triangleList
    size = img.shape
    r = (0, 0, size[1], size[0])

    for t in triangleList:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3):
            cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)
            triangle_indices.append([points.index(pt1), points.index(pt2), points.index(pt3)])
    return triangle_indices

if __name__ == '__main__':

    # Define window names
    win_delaunay = "Delaunay Triangulation"

    inputimg = sys.argv[1]
    # Read in the image
    img = cv2.imread(inputimg);

    # Keep a copy around
    img_orig = img.copy();

    # Rectangle to be used with Subdiv2D
    size = img.shape
    rect = (0, 0, size[1], size[0])

    # Create an instance of Subdiv2D
    subdiv = cv2.Subdiv2D(rect);

    # Create an array of points
    points = [];

    # Read in the points from a text file
    with open(inputimg + ".txt") as file:
        for line in file:
            x, y = line.split()
            points.append((int(x), int(y)))

    # Insert points into subdiv
    for p in points:
        subdiv.insert(p)

    # Draw delaunay triangles
    triangle_indices = draw_delaunay( img, subdiv, (255, 255, 255), points);
    filename = open(inputimg.split('.')[0] + '_tri.txt','w')

    for l in triangle_indices:
        format_l = ' '.join(map(str, l))
        filename.write("%s\n" % format_l)

    # Draw points
    for p in points:
        draw_point(img, p, (0,0,255))

    # Show results
    cv2.imshow(win_delaunay,img)
    cv2.waitKey(0)
    cv2.imwrite(inputimg.split('.')[0] + '_tri.png', img)
