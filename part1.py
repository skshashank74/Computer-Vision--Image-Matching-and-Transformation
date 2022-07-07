import cv2
import numpy as np
#import matplotlib.pyplot as plt
#from scipy.spatial import distance
from sklearn.cluster import AgglomerativeClustering
import os
import glob
import sys


#path = '/content/drive/MyDrive/CV Assignment 2/part1-images'
#jpg_files = glob.glob(os.path.join(path, "*.jpg"))
jpg_files = sys.argv[3:-1]

m = int(sys.argv[2])

def extracting_label(files):
  label = []
  for i in files:
    head, tail = os.path.split(i)
    fname = tail.split('.')[0].split('_')[0] if tail and tail.split('.') and tail.split('.')[0].split('_') else ''
    label.append(fname)
  return label

label = extracting_label(jpg_files)

def BF_matcher_KNN(img1, img2):
    # Initiate ORB detector
    orb = cv2.ORB_create(100)
    # find the keypoints and descrbatchDistanceiptors with ORB

    (keypoints1, descriptors1) = orb.detectAndCompute(img1,None)
    (keypoints2, descriptors2) = orb.detectAndCompute(img2,None)
    bf = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(descriptors1, descriptors2, k = 2)

    correct_match = []
    for m,n in matches:
        if m.distance/n.distance < 0.75:
            correct_match.append(m.distance)
    
    return sum(correct_match)


# creating distance matrix
jpg_files_1 = len(jpg_files) 
distance_matrix = np.zeros((jpg_files_1, jpg_files_1))

for i in range(jpg_files_1):
  for j in range(jpg_files_1):
    img1 = cv2.imread(jpg_files[i])
    img2 = cv2.imread(jpg_files[j])
    correct_match = BF_matcher_KNN(img1, img2)
    distance_matrix[i,j] = correct_match


cluster = AgglomerativeClustering(n_clusters=m, affinity='precomputed', linkage='average')
pred = cluster.fit_predict(distance_matrix)


len_label = len(label)
mat1 =  np.zeros((len_label,len_label))
for i in range(pred.shape[0]):
  for j in range(pred.shape[0]):
    if pred[i] == pred[j]:
      mat1[i,j] = 1

mat =  np.zeros((len_label,len_label))
for i in range(len_label):
  for j in range(len_label):
    if label[i] == label[j]:
      mat[i,j] = 1

accuracy = (np.sum(mat == mat1)-len_label)/(len_label*(len_label-1))
print("The accuracy of the clustering is", round(accuracy,3))


jpg_file_tail = []
for i in jpg_files:
  head, tail = os.path.split(i)
  jpg_file_tail.append(tail)

sorting_img = {}
for i in range(len(pred)):
  if pred[i] not in sorting_img:
    sorting_img[pred[i]] = [jpg_file_tail[i]]
  else:
    sorting_img[pred[i]].append(jpg_file_tail[i])

textfile = open(sys.argv[-1], "w")
for i in range(m):
  a = " ".join(sorting_img[i]) + '\n'
  textfile.write(a)
textfile.close()



# Own BFmatcher and visualization code (Not used)

#img1 = cv2.imread("/content/drive/MyDrive/CV Assignment 2/part1-images/eiffel_19.jpg", cv2.IMREAD_GRAYSCALE)
#img2 = cv2.imread("/content/drive/MyDrive/CV Assignment 2/part1-images/eiffel_18.jpg", cv2.IMREAD_GRAYSCALE)

def BF_matcher_visualition(img1, img2):

  orb = cv2.ORB_create(nfeatures=500)

  (keypoints1, descriptors1) = orb.detectAndCompute(img1, None)
  (keypoints2, descriptors2) = orb.detectAndCompute(img2, None)

  lookup = []
  for i in range(0, len(keypoints1)):
    temp = []
    for j in range(0, len(keypoints2)):
      # hamming
      dist = [distance.hamming(descriptors1[i],descriptors2[j]), 
              int(keypoints2[j].pt[0]), int(keypoints2[j].pt[1])]
      temp.append(dist)
    temp.sort()
    min_dist = temp[0]
    min2_dist = temp[1]
    dist_mea = min_dist[0]/min2_dist[0]
    lookup.append([dist_mea, int(keypoints1[i].pt[0]),int(keypoints1[i].pt[1]), 
                  min_dist[1], min_dist[2]])

  rows1 = img1.shape[0]
  cols1 = img1.shape[1]
  rows2 = img2.shape[0]
  cols2 = img2.shape[1]

  out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')
  out[:rows1,:cols1] = np.dstack([img1, img1, img1])
  out[:rows2,cols1:] = np.dstack([img2, img2, img2])

  image1_cord = []
  image2_cord = []
  for i in lookup:
    if i[0] < 0.85:
      cv2.circle(out, (i[1],i[2]), 4, (255, 0, 0), 1)  
      image1_cord.append((i[1],i[2]))
      cv2.circle(out, (i[3]+cols1,i[4]), 4, (255, 0, 0), 1)
      image2_cord.append((i[3],i[4]))

      cv2.line(out, (i[1],i[2]), (i[3]+cols1,i[4]), (255,255,255), 1)

  cv2_imshow(out)


def ORB_matcher(img1, img2):
  orb = cv2.ORB_create(nfeatures=500)

  (keypoints1, descriptors1) = orb.detectAndCompute(img1, None)
  (keypoints2, descriptors2) = orb.detectAndCompute(img2, None)
  lookup = []
  for i in range(0, len(keypoints1)):
    temp = []
    for j in range(0, len(keypoints2)):
      # hamming
      dist = [distance.hamming(descriptors1[i],descriptors2[j])]
      temp.append(dist)
    temp.sort()
    min_dist = temp[0]
    min2_dist = temp[1]
    dist_mea = min_dist[0]/min2_dist[0]
    lookup.append([min_dist[0], dist_mea])
  return lookup

  











