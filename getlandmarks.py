from facepp import API, File
from pprint import pformat
import sys
import cv2

files = []
for arg in sys.argv[1:]:
    files.append(arg)

api_server_international = 'https://api-us.faceplusplus.com/facepp/v3/'

API_KEY = 'TKc09eEMggaDnrTMaJ4wueum7ljWa5Lf'
API_SECRET = '3gX8LSHV7X1R4ODVxsMO-inu4qZDyOSJ'

api = API(API_KEY, API_SECRET, srv=api_server_international)

# api.faceset.delete(outer_id='test1', check_empty=0)

ret = api.faceset.create(outer_id='test1')

for filename in files:
    img = cv2.imread(filename);
    size = img.shape

    res = api.detect(image_file=File(filename), return_landmark=2)
    face = res["faces"][0]["landmark"]
    person_file = open(filename+".txt", "w")
    for face_area, face_points in face.iteritems():
        person_file.write(str(face_points[u'x']) + "\t" + str(face_points[u'y']) + "\n")

api.faceset.delete(outer_id='test1', check_empty=0)
