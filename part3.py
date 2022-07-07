import cv2
from itertools import combinations
import random
import numpy as np
from a2 import tranformation_matrix

def ORB(img1, img2):
    # convert images to grayscale
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # create SIFT object
    orb = cv2.ORB_create()
    # detect SIFT features in both images
    keypoints_1, descriptors_1 = orb.detectAndCompute(img1,None)
    keypoints_2, descriptors_2 = orb.detectAndCompute(img2,None)

    # create feature matcher
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    # match descriptors of both images
    matches = bf.match(descriptors_1,descriptors_2)

    # sort matches by distance
    matches = sorted(matches, key = lambda x:x.distance)
    # draw first 50 matches
    matched_img = cv2.drawMatches(img1, keypoints_1, img2, keypoints_2, matches[:50], img2, flags=2)

    # # show the image
    # cv2.imshow('image', matched_img)
    # # save the image
    # cv2.imwrite("matched_images.jpg", matched_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    common_points = []
    for match in matches:
            x1y1 = keypoints_1[match.queryIdx].pt
            x2y2 = keypoints_2[match.trainIdx].pt
            feature = list(map(int, list(x1y1) + list(x2y2) + [match.distance]))
            common_points.append(feature)
    return common_points

def homography(poc):
    '''
    Function to calculate homography with the given point of correspondence(poc).
    '''

    A = []
    for x, y, u, v, distance in poc:
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])
    A = np.asarray(A)

    # Taking SVD
    U, S, Vh = np.linalg.svd(A)

    L = Vh[-1, :] / Vh[-1, -1]
    H = L.reshape(3, 3)

    return H
    
def RANSAC(common_points, n_best_feat = 40, max_dist= 2):
    # Get common_points from Shashank's Code
    feature_subsets = list(combinations(common_points[:n_best_feat], 4)) # create a list with combinations of 4 points
    random.shuffle(feature_subsets)

    for feature_point in feature_subsets[:4000]:
        transformation_mat = homography(feature_point) # Replace with Ujwala's Code
        
        inlier_points = []
        inlier_count = 0
        
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
            
    best_transformation_mat = homography(best_points)

    return best_transformation_mat

def stitch_image(img1, img2):
    img1 = cv2.imread("test1.png")
    img2 = cv2.imread("warped.jpg")
    new_img = np.zeros((img2.shape[0],img2.shape[1],img2.shape[1]))

    new_img = img1
    print(new_img.shape)

    for j in range(img2.shape[1]):
        for i in range(img2.shape[0]):
            if np.all((img2[i,j] == 0)) and np.all((new_img[i,j] == 0)):
                new_img[i,j] = img2[i,j]

# Newer Code
def tranformed_image(image1,image2,M):
    im1 = cv2.imread(image1)
    im2 = cv2.imread(image2)
    
    #print(im1.shape,im2.shape)
    #initial image 
#     point = np.array([[k for k in range(im2.shape[1]) for i in range(im2.shape[0])],
#                   [k for i in range(im2.shape[1]) for k in range(im2.shape[0])],
#                   [1 for i in range((im2.shape[1])*(im2.shape[0]))]])

    boundary = np.array([[k for k in range(im2.shape[1]+1) for i in range(im2.shape[0]+1)],
                  [k for i in range(im2.shape[1]+1) for k in range(im2.shape[0]+1)],
                  [1 for i in range((im2.shape[1]+1)*(im2.shape[0]+1))]])

    a,b = boundary[:,0],boundary[:,-1]
    c,d = np.array([b[0],0,1]),np.array([0,b[1],1])
    #print(a,b,c,d)
    
    boundary = np.array([a,c,b,d]).T
    #print(point)
    
    #print(point)
    
    print(boundary.shape)
    
    #transformed image - inverse warping
    boundary_tr = M @ boundary
    boundary_tr = boundary_tr/boundary_tr[2,:]
    
    col_min,col_max = min(min(boundary_tr[0,:]),0),max(max(boundary_tr[0,:]),im1.shape[1])
    row_min,row_max = min(min(boundary_tr[1,:]),0),max(max(boundary_tr[1,:]),im1.shape[0])
    print("here")
    print(col_min,col_max,row_min,row_max)
    width,height = int(round(col_max-col_min)),int(round(row_max-row_min))
   
    #print(width,height)
    #bilinear interpolation
#   new_arr = np.zeros((im_arr.shape[0],im_arr.shape[1],im_arr.shape[2]))
    new_arr = np.zeros((height,width,im2.shape[2]))
    
    print("================================",height*width)

    point = np.array([[k for k in range(im2.shape[1]) for i in range(im2.shape[0])],
                  [k for i in range(im2.shape[1]) for k in range(im2.shape[0])],
                  [1 for i in range((im2.shape[1])*(im2.shape[0]))]])
    

    point_tr = np.linalg.inv(M) @ point
    point_tr = point_tr/point_tr[2,:]
    print(point_tr.shape)
    
    k=0
    for i in range(new_arr.shape[1]):
        for j in range(new_arr.shape[0]):
            print(i,j)
            if k < 480000:
#                 print(a,b,b_c,b_f,a_c,a_f)
#                 print(new_arr.shape[1], new_arr.shape[0])
                
#             if k < height*width:
                b, a = point_tr[:, k][0] - np.floor(point_tr[:, k][0]), point_tr[:, k][1] - np.floor(point_tr[:, k][1])
                b_c, a_c = np.int64(np.ceil(point_tr[:, k][0])), np.int64(np.ceil(point_tr[:, k][1]))
                b_f, a_f = np.int64(np.floor(point_tr[:, k][0])), np.int64(np.floor(point_tr[:, k][1]))
#                 if j == 533 and i == 230:
#                     print(a,b,b_c,b_f,a_c,a_f)
#                     print(new_arr.shape[1], new_arr.shape[0])
                if 0 <= b_f and b_f < im2.shape[1] and 0<=b_c and b_c<im2.shape[1] and 0<=a_f and a_f<=im2.shape[0] and a_c>=0 and a_c<im2.shape[0]:
                    new_arr[j, i, :] = (1 - a) * (1 - b) * im2[a_f, b_f] + a * (1 - b) * im2[a_c, b_f] + a * b * \
                                       im2[a_c, b_c] + (1 - a) * b * im2[a_f, b_c]
            k+=1
            
    
    return new_arr