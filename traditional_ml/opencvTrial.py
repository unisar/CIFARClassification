import pandas as pd
import cv2
import numpy as np
from pprint import pprint
from itertools import izip
from pyspark.sql import SparkSession
import sys
from pyspark.sql.types import *
from pyspark.sql import SQLContext
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.classification import LogisticRegression, OneVsRest
from pyspark.ml.linalg import SparseVector,Vectors, VectorUDT
from pyspark.sql.functions import udf
from pyspark.ml.clustering import GaussianMixture
from pyspark.sql import SQLContext, Row
from pyspark.mllib.clustering import KMeans
from pyspark.mllib.clustering import KMeansModel
from scipy.spatial import distance
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.regression import LabeledPoint



def get_keypoint_descriptors(fileName):
    img = cv2.imread('../data/images/'+str(fileName)+'.png')
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT()
    kp, descriptors = sift.detectAndCompute(gray, None)
  
    '''
    surf = cv2.SURF()
    kp, descriptors = surf.detectAndCompute(gray, None)
    print surf.descriptorSize()
    # SURF extraction
    '''
    return descriptors

   
def assign_pooling(row, clusterCenters,pooling="max"):
    image_name = row['fileName']
    feature_matrix = np.array(row['features'])
    label = row['label']
    clusterCenters = clusterCenters.value
    model = KMeansModel(clusterCenters)
    bow = np.zeros(len(clusterCenters))
    
    for x in feature_matrix:
        k = model.predict(x)
        dist = distance.euclidean(clusterCenters[k], x)
        if pooling == "max":
            bow[k] = max(bow[k], dist)
        elif pooling == "sum":
            bow[k] = bow[k] + dist

            
    return Row(fileName=image_name, features=Vectors.dense(bow),label=label)
    #return Row(fileName=image_name, features=(bow.tolist()),label=label)

def parsePoint(label,vec):
    return LabeledPoint(label, vec)


def main():
    #initialize spark session
    spark = SparkSession\
            .builder\
            .appName("Image Classification")\
            .getOrCreate()
    sc = spark.sparkContext

    sqlContext = SQLContext(sc)

    #read file names
    fileNames = (sys.argv)
    X_train = sc.textFile(fileNames[1]).zipWithIndex().map(lambda x:(x[1],x[0]))
    y_train = sc.textFile(fileNames[2]).zipWithIndex().map(lambda x:(x[1],x[0]))
    feature_parquet_path = './features.parquet'
    test_feature_parquet_path = './test_features.parquet'
    X_test = sc.textFile(fileNames[3]).zipWithIndex().map(lambda x:(x[1],x[0].encode('utf-8')))
    y_test = sc.textFile(fileNames[4]).zipWithIndex().map(lambda x:(x[1],x[0].encode('utf-8')))


    X_y_train = X_train.join(y_train).map(lambda (index,(X,y)):(X,float(y)))
    X_y_train_features = X_y_train.map(lambda x: (Row(fileName=x[0],label=x[1], features=get_keypoint_descriptors(x[0]).tolist())))

    features = sqlContext.createDataFrame(X_y_train_features)
    features.registerTempTable("images")
    features.write.parquet(feature_parquet_path)
    #print features.collect()

    features = sqlContext.read.parquet(feature_parquet_path)
    features = features.rdd.flatMap(lambda x: x['features']).cache()
    model = KMeans.train(features, 100, maxIterations=10, initializationMode="random")

    kmeans_model_path = './kmeans-dictionary'
    model.save(sc, kmeans_model_path)
    print("Clusters have been saved as text file to %s" % kmeans_model_path)
    print("Final centers: " + str(model.clusterCenters))
    clusterCenters = model.clusterCenters
    clusterCenters = sc.broadcast(clusterCenters)

    features = sqlContext.createDataFrame(X_y_train_features)
    features_bow = features.rdd.map(lambda row:assign_pooling(row,clusterCenters=clusterCenters, pooling="max"))
    featuresSchema = sqlContext.createDataFrame(features_bow)
    featuresSchema.registerTempTable("images")
    featuresSchema =  featuresSchema.withColumn('features',featuresSchema.features.cast(VectorUDT()))
    #featuresSchemaTrain = featuresSchema.rdd.map(lambda x:parsePoint(x['label'],x['features']))
    print featuresSchema.take(1)
    print featuresSchema.show()
    
    X_y_test = X_test.join(y_test).map(lambda (index,(X,y)):(X,float(y)))
    X_y_test_features = X_y_test.map(lambda x: (Row(fileName=x[0],label=x[1], features=get_keypoint_descriptors(x[0]).tolist())))
    test_features = sqlContext.createDataFrame(X_y_test_features)
    test_features.registerTempTable("images")
    test_features.write.parquet(test_feature_parquet_path)

    print test_features.take(1)
    test_features_bow = test_features.rdd.map(lambda row:assign_pooling(row,clusterCenters=clusterCenters, pooling="max"))
    test_featuresSchema = sqlContext.createDataFrame(test_features_bow)
    test_featuresSchema.registerTempTable("testimages")
    test_featuresSchema =  test_featuresSchema.withColumn('features',test_featuresSchema.features.cast(VectorUDT()))
    #featuresSchemaTest = test_featuresSchema.rdd.map(lambda x:parsePoint(x['label'],x['features']))

    #svm = SVMWithSGD.train(trainingData, iterations=10)
    #rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=5)
    lr = LogisticRegression(maxIter=10, regParam=0.1)
    ovr = OneVsRest(classifier=lr)
    model = ovr.fit(featuresSchema)
    predictions = model.transform(test_featuresSchema)

   
    # Select example rows to display.
    print predictions.show()
    print "predictions!!!"

    # Select (prediction, true label) and compute test error
    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test Error = %g" % (1.0 - accuracy))
    spark.stop()

 
if __name__ == "__main__":
    main()      


