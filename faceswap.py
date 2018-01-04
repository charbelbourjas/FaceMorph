import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from triangulation import draw_delaunay, draw_point, rect_contains


def read_points(person_file):
    points = [];
    with open(person_file) as file :
        for line in file :
            x, y = line.split()
            points.append((int(x), int(y)))
    return points

def triangulation(img, points, output_file_name):
    # Define window names
    win_delaunay = "Delaunay Triangulation"

    # Keep a copy around
    img_orig = img.copy();

    # Rectangle to be used with Subdiv2D
    size = img.shape
    rect = (0, 0, size[1], size[0])

    # Create an instance of Subdiv2D
    subdiv = cv2.Subdiv2D(rect);

    # Insert points into subdiv
    for p in points:
        subdiv.insert(p)

    # Draw delaunay triangles
    triangle_indices = draw_delaunay( img, subdiv, (255, 255, 255), points)


    # Draw points
    for p in points:
        draw_point(img, p, (0,0,255))

    # Show results
    cv2.imshow(win_delaunay,img)
    cv2.waitKey(0)
    cv2.imwrite(output_file_name, img)
    return triangle_indices

def warp_triangle(source_img, morphed_img, source_tri, morphed_tri):

    # Calculate bounding boxes aroung triangles (warp only small part of image)
    print "source_tri:   " + str(source_tri)
    print "morphed_tri:   " + str(morphed_tri)

    r1 = cv2.boundingRect(np.float32([source_tri]))
    r2 = cv2.boundingRect(np.float32([morphed_tri]))

    print "r1:   " + str(r1)
    print "r2:   " + str(r2)

    # Need to get location of triangles in their respective rectangles
    # subtract x and y coordinates of triangles from the bounding box coordinates
    source_tri_crop = []
    morphed_tri_crop = []

    for i in xrange(0,3):
        source_tri_crop.append(((source_tri[i][0] - r1[0]),(source_tri[i][1] - r1[1])))
        morphed_tri_crop.append(((morphed_tri[i][0] - r2[0]),(morphed_tri[i][1] - r2[1])))

    print "new_t1:   " + str(source_tri_crop)
    print "new_t2:   " + str(morphed_tri_crop)


    # Apply warpImage to small rectangular patches
    morphed_rect =  morphed_img[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    # Estimate Affine transform given pair of triangles
    warpMat = cv2.getAffineTransform( np.float32(morphed_tri_crop), np.float32(source_tri_crop) )
    source_rect = cv2.warpAffine( morphed_rect, warpMat, (r1[2], r1[3]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    # Mask pixels outside triangles
    mask = np.zeros((r1[3], r1[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(source_tri_crop), (1.0, 1.0, 1.0), 16, 0);

    # Apply mask to cropped regions
    source_rect = source_rect * mask

    # Copy triangular region of the rectangular patch to the output image
    source_img[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]] = source_img[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]] * ( (1.0, 1.0, 1.0) - mask )
    source_img[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]] = source_img[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]] + source_rect

if __name__ == '__main__' :

    # Read images
    person1_file = sys.argv[1]
    person2_file = sys.argv[2]

    img1 = cv2.imread(person1_file);
    triangulation_img = np.copy(img1)
    superimposed_img = np.copy(img1)
    img2 = cv2.imread(person2_file);
    img2_copy = np.copy(img2);
    triangulation_img2 = np.copy(img2)

    # Read array of corresponding points
    points1 = read_points(person1_file + '.txt')
    points2 = read_points(person2_file + '.txt')

    # Find convex hull
    hull1 = []
    hull2 = []

    index = cv2.convexHull(np.array(points2), returnPoints = False)

    for i in xrange(0, len(index)):
        hull1.append(points1[int(index[i])])
        hull2.append(points2[int(index[i])])

    x_1, y_1 = zip(*points1)
    x_2, y_2 = zip(*points2)

    plt.subplot(1,2,1)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.scatter(x_1,y_1,  s= 3, color='red')
    plt.subplot(1,2,2)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.scatter(x_2,y_2,  s= 3, color='red')
    # plt.show()

    # Find delanauy traingulation for convex hull points
    dt = triangulation(triangulation_img, hull1, 'triangulated_face.jpg')
    temp = triangulation(triangulation_img2, hull2, 'triangulated_morphed_face.jpg')

    if len(dt) == 0:
        quit()

    # Apply affine transformation to Delaunay triangles
    for i in xrange(0, len(dt)):
        t1 = []
        t2 = []

        #get points for img1, img2 corresponding to the triangles
        for j in xrange(0, 3):
            t1.append(hull1[dt[i][j]])
            t2.append(hull2[dt[i][j]])
        warp_triangle(superimposed_img, img2, t1, t2)

    cv2.imshow("Morphed Face", np.uint8(superimposed_img))
    cv2.waitKey(0)
    cv2.imwrite('morphedface.jpg', np.uint8(superimposed_img))

    # Calculate rough mask around the morphed face
    hull8U = []
    for i in xrange(0, len(hull1)):
        hull8U.append((hull1[i][0], hull1[i][1]))

    mask = np.zeros(img1.shape, dtype = img1.dtype)
    cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))
    # cv2.imshow("Mask", mask)
    # cv2.waitKey(0)
    r = cv2.boundingRect(np.float32(hull1))
    center = ((r[0]+int(r[2]/2), r[1]+int(r[3]/2)))
    output = cv2.seamlessClone(superimposed_img, img1 , mask, center, cv2.NORMAL_CLONE)

    cv2.imshow("Face Swapped", output)
    cv2.waitKey(0)
    cv2.imwrite('swapped_face.jpg', output)
