# load the libraries
import sys
import numpy as np
import cv2
import numpy as np
#import matplotlib.pyplot as plt
#from scipy.spatial import distance
from sklearn.cluster import AgglomerativeClustering
import os
import glob
import sys
from itertools import combinations



def tranformed_image(img_loc, M):
    im_arr = cv2.imread(img_loc)
    # M = np.array([[0.907,0.258,-182],[-0.153,1.44,58],[-0.000306,0.000731,1]])

    # initial image
    point = np.array([[k for k in range(im_arr.shape[1]) for i in range(im_arr.shape[0])],
                      [k for i in range(im_arr.shape[1]) for k in range(im_arr.shape[0])],
                      [1 for i in range(im_arr.shape[1] * im_arr.shape[0])]])

    point_tr = np.linalg.inv(M) @ point
    point_tr = point_tr / point_tr[2, :]

    # bilinear interpolation
    new_arr = np.zeros((im_arr.shape[0], im_arr.shape[1], im_arr.shape[2]))
    k = 0
    for i in range(im_arr.shape[1]):
        for j in range(im_arr.shape[0]):

            b, a = point_tr[:, k][0] - np.floor(point_tr[:, k][0]), point_tr[:, k][1] - np.floor(point_tr[:, k][1])
            b_c, a_c = np.int64(np.ceil(point_tr[:, k][0])), np.int64(np.ceil(point_tr[:, k][1]))
            b_f, a_f = np.int64(np.floor(point_tr[:, k][0])), np.int64(np.floor(point_tr[:, k][1]))

            if 0 <= b_f and b_c < im_arr.shape[1] and 0 <= a_f and a_c < im_arr.shape[0]:
                new_arr[j, i, :] = (1 - a) * (1 - b) * im_arr[a_f, b_f] + a * (1 - b) * im_arr[a_c, b_f] + a * b * \
                                   im_arr[a_c, b_c] + (1 - a) * b * im_arr[a_f, b_c]

            k += 1

    return new_arr

def tranformation_matrix(n,image_points):
    # img1_x1, img1_y1, img1_x2, img1_y2, img1_x3, img1_y3, img1_x4, img1_y4 = 141, 131, 480, 159, 493, 630, 64, 601
    # img2_x1, img2_y1, img2_x2, img2_y2, img2_x3, img2_y3, img2_x4, img2_y4 = 318, 256, 534, 372, 316, 670, 73, 473
    #print(image_points)

    def img_pts(u):
        return [int(each) for each in image_points[u].split(",")]

    if n == 1:
        img2_x1, img2_y1 = img_pts(0)[0], img_pts(0)[1]
        img1_x1, img1_y1 = img_pts(1)[0], img_pts(1)[1]

    elif n == 2:
        img2_x1, img2_y1 = img_pts(0)[0], img_pts(0)[1]
        img1_x1, img1_y1 = img_pts(1)[0], img_pts(1)[1]
        img2_x2, img2_y2 = img_pts(2)[0], img_pts(2)[1]
        img1_x2, img1_y2 = img_pts(3)[0], img_pts(3)[1]

    elif n == 3:
        img2_x1, img2_y1 = img_pts(0)[0], img_pts(0)[1]
        img1_x1, img1_y1 = img_pts(1)[0], img_pts(1)[1]
        img2_x2, img2_y2 = img_pts(2)[0], img_pts(2)[1]
        img1_x2, img1_y2 = img_pts(3)[0], img_pts(3)[1]
        img2_x3, img2_y3 = img_pts(4)[0], img_pts(4)[1]
        img1_x3, img1_y3 = img_pts(5)[0], img_pts(5)[1]

    elif n == 4:

        img2_x1, img2_y1 = img_pts(0)[0], img_pts(0)[1]
        img1_x1, img1_y1 = img_pts(1)[0], img_pts(1)[1]
        img2_x2, img2_y2 = img_pts(2)[0], img_pts(2)[1]
        img1_x2, img1_y2 = img_pts(3)[0], img_pts(3)[1]
        img2_x3, img2_y3 = img_pts(4)[0], img_pts(4)[1]
        img1_x3, img1_y3 = img_pts(5)[0], img_pts(5)[1]
        img2_x4, img2_y4 = img_pts(6)[0], img_pts(6)[1]
        img1_x4, img1_y4 = img_pts(7)[0], img_pts(7)[1]

        # img2_x1, img2_y1 = image_points[0].split(",")
        # img1_x1, img1_y1 = image_points[1].split(",")
        # img2_x2, img2_y2 = image_points[2].split(",")
        # img1_x2, img1_y2 = image_points[3].split(",")
        # img2_x3, img2_y3 = image_points[4].split(",")
        # img1_x3, img1_y3 = image_points[5].split(",")
        # img2_x4, img2_y4 = image_points[6].split(",")
        # img1_x4, img1_y4 = image_points[7].split(",")

    # k = 0
    # for i in range(1, n + 1):
    #     # exec('{} = {}'.format(getStr(2, 'x', i), image_points[k][0]))
    #     # exec('{} = {}'.format(getStr(2, 'y', i), image_points[k][1]))
    #     # exec('{} = {}'.format(getStr(1, 'x', i), image_points[k + 1][0]))
    #     # exec('{} = {}'.format(getStr(1, 'x', i), image_points[k + 1][0]))
    #     #print(getStr(2, 'x', i),getStr(2, 'y', i),getStr(2, 'x', i),getStr(2, 'x', i))
    #     # print(image_points[k].split(",")[0],image_points[k].split(",")[1])
    #     # print(getStr(2, 'x', i),getStr(2, 'y', i),getStr(1, 'x', i),getStr(1, 'y', i))
    #
    #     exec('{} = {}'.format(getStr(2, 'x', i), image_points[k].split(",")[0]))
    #     exec('{} = {}'.format(getStr(2, 'y', i), image_points[k].split(",")[1]))
    #     exec('{} = {}'.format(getStr(1, 'x', i), image_points[k+1].split(",")[0]))
    #     exec('{} = {}'.format(getStr(1, 'y', i), image_points[k+1].split(",")[1]))
    #
    #
    #
    #     k += 2

    if n == 1:
        M = np.array([[1, 0, img2_x1 - img1_x1],
                      [0, 1, img2_y1 - img1_y1],
                      [0, 0, 1]])

    elif n == 2:
        f_coord = np.array([img2_x1, img2_y1, img2_x2, img2_y2])
        i_coord = np.array([[img1_x1, -img1_y1, 1, 0],
                            [img1_y1, img1_x1, 0, 1],
                            [img1_x2, -img1_y2, 1, 0],
                            [img1_y2, img1_x2, 0, 1]])

        var = np.linalg.solve(i_coord, f_coord.T)
        M = np.array([[var[0], -var[1], var[2]],
                      [var[1], var[0], var[3]],
                      [0, 0, 1]])

    elif n == 3:
        f_coord = np.array([img2_x1, img2_y1, img2_x2, img2_y2, img2_x3, img2_y3])
        i_coord = np.array([[img1_x1, img1_y1, 1, 0, 0, 0],
                            [0, 0, 0, img1_x1, img1_y1, 1],
                            [img1_x2, img1_y2, 1, 0, 0, 0],
                            [0, 0, 0, img1_x2, img1_y2, 1],
                            [img1_x3, img1_y3, 1, 0, 0, 0],
                            [0, 0, 0, img1_x3, img1_y3, 1]])

        var = np.linalg.solve(i_coord, f_coord.T)
        M = np.array([[var[0], var[1], var[2]],
                      [var[3], var[4], var[5]],
                      [0, 0, 1]])

    elif n == 4:
        f_coord = np.array([img2_x1, img2_y1, img2_x2, img2_y2, img2_x3, img2_y3, img2_x4, img2_y4])
        i_coord = np.array([[img1_x1, img1_y1, 1, 0, 0, 0, -img2_x1 * img1_x1, -img2_x1 * img1_y1],
                            [0, 0, 0, img1_x1, img1_y1, 1, -img2_y1 * img1_x1, -img2_y1 * img1_y1],
                            [img1_x2, img1_y2, 1, 0, 0, 0, -img2_x2 * img1_x2, -img2_x2 * img1_y2],
                            [0, 0, 0, img1_x2, img1_y2, 1, -img2_y2 * img1_x2, -img2_y2 * img1_y2],
                            [img1_x3, img1_y3, 1, 0, 0, 0, -img2_x3 * img1_x3, -img2_x3 * img1_y3],
                            [0, 0, 0, img1_x3, img1_y3, 1, -img2_y3 * img1_x3, -img2_y3 * img1_y3],
                            [img1_x4, img1_y4, 1, 0, 0, 0, -img2_x4 * img1_x4, -img2_x4 * img1_y4],
                            [0, 0, 0, img1_x4, img1_y4, 1, -img2_y4 * img1_x4, -img2_y4 * img1_y4]])

        var = np.linalg.solve(i_coord, f_coord.T)
        M = np.array([[var[0], var[1], var[2]],
                      [var[3], var[4], var[5]],
                      [var[6], var[7], 1]])

    return M

# def getStr(image_num,axis, coord):
#     return 'img' + str(image_num) + '_' + str(axis) + str(coord)



## Part 1
# extracting label from the file
def extracting_label(files):
  label = []
  for i in files:
    head, tail = os.path.split(i)
    fname = tail.split('.')[0].split('_')[0] if tail and tail.split('.') and tail.split('.')[0].split('_') else ''
    label.append(fname)
  return label

# BF matcher function
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

  
# Part 3

# Finding matching points using ORB
def ORB(img1, img2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    orb = cv2.ORB_create()
    
    keypoints_1, descriptors_1 = orb.detectAndCompute(img1,None)
    keypoints_2, descriptors_2 = orb.detectAndCompute(img2,None)

    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    matches = bf.match(descriptors_1,descriptors_2)
    matches = sorted(matches, key = lambda x:x.distance)
    
    matched_img = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:50], img2, flags=2)

    common_points = []
    for match in matches:
            x1y1 = keypoints_1[match.queryIdx].pt
            x2y2 = keypoints_2[match.trainIdx].pt
            feature = list(map(int, list(x1y1) + list(x2y2) + [match.distance]))
            common_points.append(feature)
    return common_points

# Find tranformation matrix
def trans_matrix(image_points):

    i_coord = []
    for img1_x1, img1_y1, img2_x1, img2_y1, dist in image_points:
        i_coord.append([img1_x1, img1_y1, 1, 0, 0, 0, -img2_x1*img1_x1, -img2_x1*img1_y1, -img2_x1])
        i_coord.append([0, 0, 0, img1_x1, img1_y1, 1, -img2_y1*img1_x1, -img2_y1*img1_y1, -img2_y1])
    i_coord = np.asarray(i_coord)

    # Taking SVD
    A, B, points = np.linalg.svd(i_coord)
    
    temp_mat = points[-1, :] / points[-1, -1]
    M = temp_mat.reshape(3, 3)

    return M

# Implementation of RANSAC
def RANSAC(common_points, n_best_feat = 40, max_dist= 4):
    feature_subsets = list(combinations(common_points[:n_best_feat], 4))
    best_count = 0
    best_points = 0

    for feature_point in feature_subsets[:5000]:
        inlier_points = []
        inlier_count = 0
        transformation_mat = trans_matrix(feature_point)
        
        for point in common_points[:n_best_feat]:
            initial_point = np.array(point[:2] + [1]).reshape(-1,1)
            final_point = np.array(point[2:4] + [1]).reshape(-1,1)
            
            mapped_point = np.dot(transformation_mat, initial_point)
            mapped_point/= mapped_point[-1,-1]
            distance = np.linalg.norm(mapped_point - final_point)
            
            if distance < max_dist:
                inlier_count+=1
                inlier_points.append(point)
            
        if inlier_count > best_count:
            best_points = inlier_points
            best_count = inlier_count
            
    return trans_matrix(best_points)

# Warping Image
def image_transform(im_arr,M):

    point = np.array([[k for k in range(im_arr.shape[1]+1) for i in range(im_arr.shape[0]+1)],
                  [k for i in range(im_arr.shape[1]+1) for k in range(im_arr.shape[0]+1)],
                  [1 for i in range((im_arr.shape[1]+1)*(im_arr.shape[0]+1))]])
    
    a,b = point[:,0],point[:,-1]
    c,d = np.array([b[0],0,1]),np.array([0,b[1],1])
    boundary = np.array([a,c,b,d]).T
    
    boundary_tr = M @ boundary
    boundary_tr = boundary_tr/boundary_tr[2,:]
    
    col_min,col_max = min(min(boundary_tr[0,:]),0),max(max(boundary_tr[0,:]),im_arr.shape[1])
    row_min,row_max = min(min(boundary_tr[1,:]),0),max(max(boundary_tr[1,:]),im_arr.shape[0])
    
    col_offset = 0 - col_min
    row_offset = 0 - row_min

    width,height = int(round(col_max-col_min)),int(round(row_max-row_min))
    new_arr = np.zeros((height,width,3))
    
    M[:,2] = M @ np.array([-int(col_min),-int(row_min),1]).T
    
    point = np.array([[k for k in range(new_arr.shape[1]) for i in range(new_arr.shape[0])],
                  [k for i in range(new_arr.shape[1]) for k in range(new_arr.shape[0])],
                  [1 for i in range((new_arr.shape[1])*(new_arr.shape[0]))]])
    
    point_tr = np.linalg.inv(M) @ point
    point_tr = point_tr/point_tr[2,:]
    
    k=0
    for col in range(new_arr.shape[1]):
        for row in range(new_arr.shape[0]):

            b,a = point_tr[:,k][0] - np.floor(point_tr[:,k][0]),point_tr[:,k][1] - np.floor(point_tr[:,k][1])
            b_c,a_c = np.int64(np.ceil(point_tr[:,k][0])),np.int64(np.ceil(point_tr[:,k][1]))
            b_f,a_f = np.int64(np.floor(point_tr[:,k][0])),np.int64(np.floor(point_tr[:,k][1]))

            k+=1

            if (a_c>=0 and a_c<im_arr.shape[0] and a_f>=0 and a_f<im_arr.shape[0] and b_c>=0 and b_c<im_arr.shape[1] and b_f>=0 and b_f<im_arr.shape[1] ):

                new_arr[row,col,:] = (1-b)*(1-a)*im_arr[a_f,b_f] +(1-b)*(a)*im_arr[a_c,b_f]+b*(1-a)*im_arr[a_f,b_c] +b*a*im_arr[a_c,b_c]
    
    return int(row_offset), int(col_offset), new_arr

def create_panorama(file_path, row_offset, col_offset, img1, img2):
    new_img = np.zeros((img2.shape[0],img2.shape[1],img2.shape[2]))
    new_img[row_offset:row_offset+img1.shape[0], col_offset:col_offset+img1.shape[1]] = img1
    new_img = new_img.astype(int)

    for j in range(img2.shape[1]):
        for i in range(img2.shape[0]):
            if not np.all((img2[i,j] == 0)) and np.all((new_img[i,j] == 0)):
                new_img[i,j] = img2[i,j]
    cv2.imwrite(file_path, new_img)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    if sys.argv[1] == 'part1':

        m = int(sys.argv[2])
        jpg_files = sys.argv[3:-1]
        label = extracting_label(jpg_files)
        print("Looking for matches!!!!!")

        # creating distance matrix
        jpg_files_1 = len(jpg_files) 
        distance_matrix = np.zeros((jpg_files_1, jpg_files_1))
        for i in range(jpg_files_1):
          for j in range(jpg_files_1):
            img1 = cv2.imread(jpg_files[i])
            img2 = cv2.imread(jpg_files[j])
            correct_match = BF_matcher_KNN(img1, img2)
            distance_matrix[i,j] = correct_match

        # Agglomerative Clustering
        cluster = AgglomerativeClustering(n_clusters=m, affinity='precomputed', linkage='average')
        pred = cluster.fit_predict(distance_matrix)

        # label comparision
        len_label = len(jpg_files)
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

        # assigning to a label
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
        print("Clustered images files has been generated!!!")


    if sys.argv[1] == 'part2':
        n = int(sys.argv[2])
        dest_image = sys.argv[3]
        source_image = sys.argv[4]
        image_output = sys.argv[5]
        image_points = sys.argv[6:]
        #print(image_points)
        M = tranformation_matrix(n,image_points)
        new_arr = tranformed_image(source_image, M)
        cv2.imwrite(image_output, new_arr)

    if sys.argv[1] == 'part3':
      im1_name = sys.argv[2]
      im2_name = sys.argv[3]
      file_name = sys.argv[4]

      img1 = cv2.imread(im1_name)
      img2 = cv2.imread(im2_name)

      common_points = ORB(img1, img2)
      print("Common points found!")
      transformation_matrix = RANSAC(common_points)
      row_offset, col_offset, warped_img = image_transform(img1, transformation_matrix)
      print("Image has been warped!")
      create_panorama(file_name, row_offset, col_offset, img2, warped_img)
      print("Output file is created successfully!")

# Our implementation of BF matcher and visualization

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

