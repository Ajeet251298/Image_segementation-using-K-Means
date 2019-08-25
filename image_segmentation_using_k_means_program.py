import matplotlib.pyplot as plt

import cv2

im =  cv2.imread('elephant.jpg') #Reads an image into BGR Format

im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB) #change from BGR to RGB
original_shape = im.shape
print(im.shape)

plt.imshow(im) # show as RGB Format
plt.show()

# Flatten Each channel of the Image
all_pixels  = im.reshape((-1,3)) # for each channel we will have one liner array (means linear array for each color in cluster)
print(all_pixels.shape)

#We are using predefined sklearn for kmeans bcos sklearn uses best way of initilisation which means alot.
from sklearn.cluster import KMeans

dominant_colors = 4

km = KMeans(n_clusters=dominant_colors) #no. of cluster will be equal to colors

#fit fun is going to get cluste for each type of color in 3d RBG co-ordinate
km.fit(all_pixels) #provide all pixels of color into algorithm

centers = km.cluster_centers_ #centers are the RGB value of the color
print(centers)# in float

centers = np.array(centers,dtype='uint8')#change into int
print(centers)

###plot what color are these
#using matplotlib plotting 1*4 subplot
i = 1

plt.figure(0,figsize=(8,2))


colors = []

for each_col in centers:
    plt.subplot(1,4,i)   #1*4 
    plt.axis("off")
    i+=1
    
    colors.append(each_col)
    
    #Color Swatch
    a = np.zeros((100,100,3),dtype='uint8') #we are making 3d matrix then after we will asign corresponding color value in that matrix
    a[:,:,:] = each_col 
    plt.imshow(a)
    
plt.show()

###segmenting our main image

new_img = np.zeros((330*500,3),dtype='uint8')  #creating matrix of same shape which had origional flatterened image

print(new_img.shape)

colors#isme 4 color cluster  k center h

km.labels_  #it will give lables. mtlb kis grid m kaun sa color hai.

for ix in range(new_img.shape[0]):
    new_img[ix] = colors[km.labels_[ix]]   #we are extracting level for every pixel 
    
new_img = new_img.reshape((original_shape))
plt.imshow(new_img)
plt.show()















