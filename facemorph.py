import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt

def read_points(person_file):
    # Create an array of points.
    points = [];
    # Read points
    with open(person_file) as file :
        for line in file :
            x, y = line.split()
            points.append((int(x), int(y)))

    return points

def calculate_morph_img_points(person_one_points, person_two_points,a):
    morphed_points = []
    for pair_count, pair in enumerate(person_one_points):
        x_m = (1-a)*person_one_points[pair_count][0] + a*person_two_points[pair_count][0]
        # print x_m
        y_m = (1-a)*person_one_points[pair_count][1] + a*person_two_points[pair_count][1]
        morphed_points.append((x_m,y_m))
    return morphed_points

def calculate_affine_transforms(img_tri, tri, morphed_tri, size):
    M = cv2.getAffineTransform(np.float32(tri), np.float32(morphed_tri))
    dst = cv2.warpAffine( img_tri, M, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )
    return dst

def morph_triangle(img1, img2, img_morph, t1, t2, morphed_t,a):
    # Calculate boudning boxes aroung triangles (warp only small part of image)
    print "t1:   " + str(t1)
    print "t2:   " + str(t2)
    print "morphed_t:   " + str(morphed_t)


    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([morphed_t]))

    print "r1:   " + str(r1)
    print "r2:   " + str(r2)
    print "r:   " + str(r)

    # Need to get location of triangles in their respective rectangles
    # subtract x and y coordinates of triangles from the bounding box coordinates
    new_t1_coord = []
    new_t2_coord = []
    new_morphed_t_coord = []

    for i in xrange(0,3):
        new_t1_coord.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        new_t2_coord.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))
        new_morphed_t_coord.append(((morphed_t[i][0] - r[0]),(morphed_t[i][1] - r[1])))

    print "new_t1:   " + str(new_t1_coord)
    print "new_t2:   " + str(new_t2_coord)
    print "new_morphed_t:   " + str(new_morphed_t_coord)


    # Apply warpImage to small rectangular patches
    img1_rect =  img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    # print "img1_rect:   " + str(img1_rect)
    img2_rect =  img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]
    # print "img2_rect:   " + str(img2_rect)
    #
    # plt.imshow(cv2.cvtColor(img1_rect, cv2.COLOR_BGR2RGB))
    # plt.plot(new_t1_coord)
    # plt.show()

    # Estimate Affine transform given pair of triangles
    size = (r[2], r[3])
    warped_image1 = calculate_affine_transforms(img1_rect, new_t1_coord, new_morphed_t_coord,size)
    warped_image2 = calculate_affine_transforms(img2_rect, new_t2_coord, new_morphed_t_coord,size)

    # Mask pixels outside triangles
    mask = np.zeros((r[3], r[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(new_morphed_t_coord), (1.0, 1.0, 1.0), 16, 0);

    # Alpha blend rectangular patches
    imgRect = (1.0 - a) * warped_image1 + a * warped_image2

    # Copy triangular region of the rectangular patch to the output image
    img_morph[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img_morph[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( 1 - mask ) + imgRect * mask

# Find location of feature points in morphed image
person1_file = sys.argv[1]
person2_file = sys.argv[2]
a = 0.5

img1 = cv2.imread(person1_file)
img2 = cv2.imread(person2_file)
imgMorph = np.zeros(img1.shape, dtype = img1.dtype)

points1 = read_points(person1_file + ".txt")
points2 = read_points(person2_file + ".txt")
morphed_points = calculate_morph_img_points(points1, points2, a)
# print morphed_points

x_1, y_1 = zip(*points1)
x_2, y_2 = zip(*points2)

plt.subplot(1,3,1)
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.scatter(x_1,y_1,  s= 3, color='red')
plt.subplot(1,3,2)
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plt.scatter(x_2,y_2,  s= 3, color='red')
# plt.imwrite()
plt.show()

# Read triangles from tri.txt
with open(person1_file.split('.')[0] +"_tri.txt") as file :
    for line in file :
        x,y,z = line.split()

        x = int(x)
        y = int(y)
        z = int(z)

        t1 = [points1[x], points1[y], points1[z]]
        t2 = [points2[x], points2[y], points2[z]]
        t = [ morphed_points[x], morphed_points[y], morphed_points[z] ]

        # Morph one triangle at a time.
        morph_triangle(img1, img2, imgMorph, t1, t2, t, a)

# Display Result
plt.subplot(1,3,3)
cv2.imshow("Morphed Face", np.uint8(imgMorph))
# cv2.waitKey(0)
cv2.imwrite(person1_file.split('.')[0] + person2_file.split('.')[0] + '.jpg',  np.uint8(imgMorph))
