import os
import cv2
from PIL import Image
import numpy as np
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot, pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA











def read_no_face(dir_name):
    labels_vector = np.arange(800)
    for i in range(800):
        if(i<400):
            labels_vector[i]=0
        else:
            labels_vector[i]=1
    matrix_input_no_faces = np.zeros(shape=(1, 10304))
    for subdir, dirs, files in sorted(os.walk(dir_name)):
        for filename in sorted(files, key=len):
            filepath = subdir + os.sep + filename
            if filepath.endswith(".pgm"):
                new_row = convert_img(filepath)
                matrix_input_no_faces = np.append(matrix_input_no_faces, np.matrix(new_row), axis=0)
            if filepath.endswith(".jpg"):
                img = Image.open(filepath)
                resized_img = img.resize((92,112))
                img = resized_img.convert('L')
                ready_image = np.array(img).reshape((1,10304))
                #print(ready_image)
                matrix_input_no_faces = np.append(matrix_input_no_faces, ready_image, axis=0)
    
                
                

    matrix_input_no_faces=np.delete(matrix_input_no_faces,0,0)
    #print(matrix_input_no_faces.shape)
    #print(labels_vector)
    #print(labels_vector.shape)
    return matrix_input_no_faces

def read_file(dir_name):
    labels_vector = np.arange(400)  # 40 subj 10 img each
    k = 0
    for i in range(400):
        if (i % 10 != 0):
            labels_vector[i] = k
        else:
            k = k + 1
            labels_vector[i] = k
    matrix_input = np.zeros(shape=(1, 10304))
    #   print(matrix_input)
    for subdir, dirs, files in sorted(os.walk(dir_name)):
        for filename in sorted(files, key=len):
            #   print(filename)
            filepath = subdir + os.sep + filename
            #   print(filepath)
            if filepath.endswith(".pgm"):
                # print(filepath)
                new_row = convert_img(filepath)
                matrix_input = np.append(matrix_input, np.matrix(new_row), axis=0)

    matrix_input = np.delete(matrix_input, 0, 0)
    # print(matrix_input.shape)
    return matrix_input


def convert_img(img_path):
    # print(img_path)
    img = cv2.imread(img_path, -1).flatten()
    # print(img.shape)
    return img


def data_split(imgs):
    training = np.zeros(shape=(1, 10304))
    testing = np.zeros(shape=(1, 10304))
    split_labels_vector = np.arange(200)
    k = 0
    for i in range(200):
        if (i % 5 != 0):
            split_labels_vector[i] = k
        else:
            k = k + 1
            split_labels_vector[i] = k

    for i in range(400):
        if (i % 2 == 0):
            testing = np.append(testing, np.matrix(imgs[i]), axis=0)
        else:
            training = np.append(training, np.matrix(imgs[i]), axis=0)

    testing = np.delete(testing, 0, 0)
    training = np.delete(training, 0, 0)
    return testing, training, split_labels_vector

def data_split_no_face(imgs):
    training = np.zeros(shape=(1, 10304))
    testing = np.zeros(shape=(1, 10304))
    split_labels_vector_no_face = np.arange(400)

    for i in range(400):
        if (i<200):
            split_labels_vector_no_face[i] = 0          #face
        else:
            split_labels_vector_no_face[i] = 1          #noface


    for i in range(800):
        if (i % 2 == 0):
            testing = np.append(testing, np.matrix(imgs[i]), axis=0)
        else:
            training = np.append(training, np.matrix(imgs[i]), axis=0)

    testing = np.delete(testing, 0, 0)
    training = np.delete(training, 0, 0)
    return testing, training, split_labels_vector_no_face

def PCA(training, testing, alpha, split_labels_vectors):
    print("PCA-----------------------------------------")
    mean = np.mean(training, axis=0)
    # print(mean.shape)
    centralized_matrix = training - mean
    cov = np.cov(centralized_matrix, bias=True, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    evalues_mat = np.diag(eigenvalues)
    # print("verification\n", cov - np.dot(np.dot(eigenvectors, np.diag(eigenvalues)), np.transpose(eigenvectors)))
    index = np.argsort(eigenvalues)[::-1]
    sorted_evalues = eigenvalues[index]
    sorted_evectors = eigenvectors[:, index]


    r = 0
    while (np.sum(sorted_evalues[0:r]) / np.sum(sorted_evalues) - alpha <= 1e-6):
        r += 1
    # print(r)
    projected_matrix_training = np.dot(training, sorted_evectors[:, :r])
    projected_matrix_testing = np.dot(testing, sorted_evectors[:, :r])

    Knn = [1, 3, 5, 7]

    score_calc_pca(Knn,projected_matrix_training, projected_matrix_testing, split_labels_vectors)

def lda_no_face(training_no_face, split_labels_vectors,testing_no_face):
    mean_training = np.mean(training_no_face, axis=0)

    mean = np.zeros(shape=(1, 10304))
    S = np.zeros(shape=(1, 10304))
    Sb = np.zeros(shape=(1, 10304))
    for i in range(200, len(training_no_face) + 200, 200):
        mean = np.append(mean, np.matrix(np.mean(training_no_face[i - 200:i], axis=0)), axis=0)

    mean = np.delete(mean, 0, 0)
    print(mean)
    print(mean.shape)
    for i in range(2):
        Sb = Sb + np.dot(200 * ((mean[i] - mean_training).T), mean[i] - mean_training)


    k = 0
    Z = np.zeros(shape=(1, 10304))
    for i in range(200, len(training_no_face) + 200, 200):
        Z = np.append(Z, np.matrix(training_no_face[i - 200:i]) - mean[k], axis=0)
        k = k + 1

    Z = np.delete(Z, 0, 0)
    print("Z =     ", Z)
    print("Z shape=     ", Z.shape)
    print("mean shape=     ", mean.shape)
    for i in range(10):
        S = S + (np.dot(Z[i].T, Z[i]))

    eigenvalues, eigenvectors = np.linalg.eigh(np.dot(np.linalg.inv(S), Sb))

    Knn = [1,3,5,7]
    score_calc_lda2(eigenvalues, eigenvectors, split_labels_vectors,training_no_face,testing_no_face)

def lda(training, split_labels_vectors, testing):
    print("LDA------------------------------")
    mean_training=np.mean(training,axis=0)

    mean= np.zeros(shape=(1,10304))
    S= np.zeros(shape=(1,10304))
    Sb=np.zeros(shape=(1,10304))
    for i in range(5,len(training)+5,5):
        mean=np.append(mean, np.matrix(np.mean(training[i-5:i], axis=0)), axis=0)



    mean = np.delete(mean, 0, 0)

    for i in range(40):

        Sb=Sb+np.dot(5*((mean[i]-mean_training).T), mean[i]-mean_training)

    # print(Sb)
    k=0
    Z=np.zeros(shape=(1,10304))
    for i in range(5, len(training) + 5, 5):
        Z=np.append(Z,np.matrix(training[i-5:i])-mean[k],axis=0)
        k=k+1


    Z = np.delete(Z, 0, 0)
    for i in range(200):
        S = S+(np.dot(Z[i].T, Z[i]))

    eigenvalues, eigenvectors = np.linalg.eigh(np.dot(np.linalg.inv(S), Sb))

    Knn=[1,3,5,7]
    score_calc_lda(Knn,eigenvalues, eigenvectors,split_labels_vectors,training, testing)

def score_calc_lda2(eigenvalues, eigenvectors,split_labels_vectors,training,testing):
    print("score LDA_____________________")
    index = np.argsort(eigenvalues)[::-1]
    sorted_evectors = eigenvectors[:, index]

    dims = sorted_evectors[:, 0:39]
    projected_matrix_training = np.dot(training, dims)
    projected_matrix_testing = np.dot(testing, dims)
    scores=[0,0,0,0]
    k=0
    for i in range(250,401,50):
        neigh = KNeighborsClassifier(n_neighbors=1, weights='distance')
        neigh.fit(projected_matrix_training[0:i,:], split_labels_vectors[:i])
        scores[k] = neigh.score(projected_matrix_testing, split_labels_vectors)
        k+=1
    nofacesnum=[250,300,350,400]
    plt.scatter(nofacesnum, scores)
    plt.show()

def score_calc_lda(Knn,eigenvalues, eigenvectors,split_labels_vectors,training,testing):
    print("score LDA_____________________")
    index = np.argsort(eigenvalues)[::-1]
    sorted_evectors = eigenvectors[:, index]

    dims = sorted_evectors[:, 0:39]
    projected_matrix_training = np.dot(training, dims)
    projected_matrix_testing = np.dot(testing, dims)
    scores=[0,0,0,0]
    for i in range(len(Knn)):
        neigh = KNeighborsClassifier(n_neighbors=Knn[i], weights='distance')
        neigh.fit(projected_matrix_training, split_labels_vectors)
        scores[i] = neigh.score(projected_matrix_testing, split_labels_vectors)

    plt.scatter(Knn, scores)
    plt.show()
def score_calc_pca(Knn,projected_matrix_training, projected_matrix_testing, split_labels_vectors):
    print("score-PCA------------------------------")
    scores = [0,0,0,0]
    for i in range(len(Knn)):
        neigh = KNeighborsClassifier(n_neighbors=Knn[i], weights='distance')
        neigh.fit(projected_matrix_training, split_labels_vectors)
        scores[i] = neigh.score(projected_matrix_testing, split_labels_vectors)

    plt.scatter(Knn, scores)
    plt.show()

if __name__ == "__main__":


    imgs = read_file("E:\Projects\python\pattern1\cv2")
    imgs_no_face=read_no_face("C:/Users/seife/Downloads/nofaces")
    #testing, training, split_labels_vectors = data_split(imgs)
    testing_no_face, training_no_face, split_labels_vector_no_face= data_split_no_face(imgs_no_face)
    # ALPHA = [0.8, 0.85, 0.9, 0.95]
    print(imgs_no_face)
    lda_no_face(training_no_face, split_labels_vector_no_face,testing_no_face)
    # PCA(training, testing, 0.95, split_labels_vectors)
    # lda(training, split_labels_vectors,testing)

