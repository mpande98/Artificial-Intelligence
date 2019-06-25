# Experiment with k means algorithm 
# Use pillow to load trees.png and read its pixel values 
# Choose 3 representative k values and use sklearn.cluster
# Take screenshots of results and explain in README
from sklearn import cluster 
from sklearn.cluster import KMeans
import numpy as np 
from PIL import Image 
import imageio 
import matplotlib.pyplot as plt

K_values = [2, 12, 36]
# Used https://www.analyticsvidhya.com/blog/2019/04/introduction-image-segmentation-techniques-python/
# for guidance on image segmentation 
def compute(image, n):

    # initialize means, pick K cluster centers 
    # pick them randomly or based on K-means++
    # convert 2d back to 3d 
    # assign each pixel in image to cluster with closest mean 
    # calculate new mean for each cluster center by averaging all pixels in cluster until convergence
    # convergence = no change in clusters OR max number of iterations reached 

    pass 

def newImage(kmeans, centroids, image):
    plt.figure()
    plt.clf()
    two_dim = []
    for i in range(len(kmeans.labels_)):  
            res = centroids[kmeans.labels_[i]]
            two_dim.append(res)
    two_dim = np.array(two_dim, dtype= np.uint8)
    return two_dim  

if __name__ == "__main__":


    # Using imageio as suggested on piazza 
    image = imageio.imread('trees.png', pilmode='RGB')
    #plt.imshow(image)

    # convert to shape that is length*width, 3 
    image2d = image.reshape(-1,3)
    #print(image2d)
    
    for n in K_values: 
        kmeans = KMeans(n_clusters = n, random_state=0).fit(image2d)
        centroids = kmeans.cluster_centers_

        two_dim = newImage(kmeans, centroids, image)
        three_dim = two_dim.reshape(image.shape)
        plt.imshow(three_dim)
        #labels = kmeans.predict(image2d)
        #print(labels)
        #print(kmeans.labels_)

        # kmeans_labels tells us which pixel of the image belongs to which cluster 
        # kmeans_labels is an array   
        """for i in range(len(kmeans.labels_)):  
            res = centroids[kmeans.labels_[i]]
            two_dim.append(res)
        
        two_dim = np.array(two_dim, dtype= np.uint8)
        three_dim = two_dim.reshape(image.shape)
        plt.imshow(three_dim)
        """
    plt.show()


