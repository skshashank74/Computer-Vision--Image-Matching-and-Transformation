# Assignment 2 

## Part 1: Image matching and clustering

This part of the assignment can be divided broadly into two parts:-

### Extracting labels: 
As each file has different file name we wanted to extract the label of each file that was done by using "extracting label" function.

### Image Matching: 
To match the features, we first extracted the features from each of the images using ORB from cv2. Once the feature is extracted we use BF matcher function and extract top two features which has the least hamming  distance. We calculated the ratio of minimum and 2nd minimum distance and took a cut off of 0.75. After that we summed all the distance for the two pairs of images to get the distance between two images. We calculate the distance between each images and stored the value in a 2D numpy matrix. In our example of 93 images, the matrix size is (93, 93)

Initially, I made my own BF matcher and visualization function. But to run for all the 93 images it took a lot of time. As BFmatcher was allowed to use we replaced our matching function with cv2 matching function. In the source code, I have added the code for the matching function.

BF matcher           |  BF matcher from scratch
:-------------------------:|:-------------------------:
![BF matcher](https://github.iu.edu/cs-b657-sp2022/sk128-ujmusku-partrao-a2/blob/main/BF%20matcher1.png)  |  ![Bf matcher scratch](https://github.iu.edu/cs-b657-sp2022/sk128-ujmusku-partrao-a2/blob/main/BF%20matcher_scratch.png)


### Clustering:
After getting the distance we ran an Agglomerative clustering using sklearn library to cluster the data. It outputs cluster number.

### Accuracy:
For calculating the accuracy we created a 2D matrix of size (93,93) for label and prediction. In the matrix, we fill the value by one when the labels are same else 0, similarly for the predicted matrix. We compare these matrix to check the matches. We get an accuracy of 81.3%

### Failures:
Even though we got decent accuracy based on the logic provided there were  cases which had different label but came in the same cluster. For example, for in one of the cluster, bigben_14.jpg and bigben_13.jpg in the same cluster but in that same cluster colosseum_5.jpg was also clustered. The feature mapped in these two images can be seen below

Same clutser same label          |  Same cluster different label
:-------------------------:|:-------------------------:
![BF matcher](https://github.iu.edu/cs-b657-sp2022/sk128-ujmusku-partrao-a2/blob/main/In%20same%20cluster.png)  |  ![Bf matcher scratch](https://github.iu.edu/cs-b657-sp2022/sk128-ujmusku-partrao-a2/blob/main/In%20same%20cluster%20different%20label.png)


## Part 2: Image transformations

There are mainly two sections to perform image transformations.
1. Based on the transformation type, output the transformation(homography matrix)
2. Warp the image based on the transformation matrix

1. Transformation Matrix:

To obtain transformation matrix, we need to understand the set of transformations that were done. For a given pair of co-ordinates in two images, we were supposed to perform translation(n=1), eucliedean(n=2), affine(n=3), and projective(n=4) transformations. In translation, we are essentially moving the objects in the first image. In Euclidean, both rotation and translation are included. In Affine, there are a series of transformations that occur i.e., rotation, scaling, sheer, and mirroring. Therefore, origin doesn't necessarily align, but parallel lines, lines, ratio, traingle mapping are all preserved. In Projective, both affine and projective warping occurs. The properties of Projective transformation are non preservation of ratios, parallelity. However, they are closed under composition, and quadrilaterals map to quadrilaterals. We'll need n pair of points to perform set of transformations(n=1,2,3,4). I solved equations using linear algebra to come up with set of matrices to solve transformation matrix for different n values. Please find the results of the images after performing set of transformations.


Translation(n=1)    |  Euclidean Rigid(n=2)  |  Affine(n=3)   |  Projective(n=4) 
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![book21](https://media.github.iu.edu/user/18351/files/4fc3bd14-4e69-4f48-86ba-fc2a7dcb9146)  |  ![book22](https://media.github.iu.edu/user/18351/files/a3d9982f-ba2e-4822-8b5d-689b78d66e32)  | ![book23](https://media.github.iu.edu/user/18351/files/9ed0e4ae-f009-48a3-96d1-b0c216818f6e) |  ![book24](https://media.github.iu.edu/user/18351/files/1841fa9a-2af4-4d31-b266-cce1ab81fdc0)

2. Warping image:

To perform warping, we'll need tranformation matrix and initial image. We obtain the transformation matrix from above. We essentially iterate over each pixel position in the initial image, perform transformation, and come up with new position of the current pixel in the warped image. After finding the new pixel location in the warped image, we place the pixel value of its initial location in the new location. I performed inverse warping with bilinear interpolation to avoid any holes in the warped image. There are obviously better interpolation methods such as bicubic interpolation for better quality of images. 

Initial Image         |  Warped Image
:-------------------------:|:-------------------------:
![lincoln](https://media.github.iu.edu/user/18351/files/0dacb9eb-7061-4797-a06b-f2fffaa067c1)  |  ![warped_linc](https://media.github.iu.edu/user/18351/files/82f815b2-7c4f-4be9-864b-bc4a213c16de)

## Part 3: Automatic Image Matching and Transformations

This task involes 4 distinct parts:
1. Extract interest points from each image
2. RANSAC to figure best relative transformation
3. Transform images to common coordinate system
4. Blend the images together

### Interest Points

We extracted feature points by using ORB implementation from OpenCV. Then we used BFMatcher to extract the top features. The resultant set of pairs of points were sorted by distance and the top 50 best mactched points were used.

### RANSAC
The RANSAC algorithm was implemented as follows:
1. The extracted common points we arranged in combinations of 4 points in order to use them as our hypothesis.
2. Iterate through the sets of 4 point pairs created and for each set (hypothesis) calculate the homography matrix.
3. Using calculated homography matrix calculate the mapped points for the remaining points and compare with the points extracted. If the mapped point falls within a threshold of a certain number of pixels with the points found through ORB, then we count it as an inlier. 
4. We keep count of total number inliers using a particular hypothesis. We also keep track of the best inliers and hypothesis.
5. Once we have iterated through all our hypothesis we calulate the best homography matrix using the best set of inlier points that we have stored.

### Image Tranformation

Once the best homography matrix is calculated we calculate the warped image transformation of the second image in the coordinate system of the first image.

### Blending

Here the two images, the original image and the warped image need to be stitched together. Our algorithm first places the original image in the canvas, and then places pixels from the warped image only where the pixel value is 0 in the final image. This way we get a fully stitched image. 

Here are some examples:

Image 1           |  Image 2           | Stitched Image           
:-------------------------:|:-------------------------:|:-------------------------:
![test1](https://media.github.iu.edu/user/18438/files/19caf5da-826a-45b0-a34f-222c996b8b7f) | ![test2](https://media.github.iu.edu/user/18438/files/1f8cfda7-9d50-438e-b780-9b85276d46c0) | ![output](https://media.github.iu.edu/user/18438/files/0260b3cd-5670-40e0-b9ec-78c938989602)



Image 1           |  Image 2           | Stitched Image           
:-------------------------:|:-------------------------:|:-------------------------:
![room1](https://media.github.iu.edu/user/18438/files/d8928016-49fd-47eb-931a-b63f5a268b49)|  ![room2](https://media.github.iu.edu/user/18438/files/3480931b-ca6f-48a7-90a8-7ce70e1e36e8)| ![output](https://media.github.iu.edu/user/18438/files/c343d16a-0ab4-4f2d-8319-3567916c5162)


### Contribution:
Brainstorming : Done by all three of us <br />
Question 1 code implementation and report: Shashank <br />
Question 2 code implementation and report: Ujwala <br />
Question 3 code implementation and report: Parth <br />

### References:
Q1: https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html <br />
    https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html <br />
Q2: https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/EPSRC_SSAZ/node11.html <br />
Q3: http://cs.brown.edu/courses/csci1290/asgn/proj5_panorama/index.html <br />





