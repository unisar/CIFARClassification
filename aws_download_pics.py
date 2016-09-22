import subprocess
import urllib

'''
download train and test images from S3 to local drive
'''

#settings:
files = ["./data/X_small_test.txt","./data/X_small_train.txt"]

#download files
filenames = []
for file in files:
    with open(file,'r') as f:
        filenames += f.readlines()

for i in xrange(len(filenames)):
    file = filenames[i].replace('\n','').strip()
    print 'Getting file %i of %i: %s' % (i+1,len(filenames),file)    
    imgopen = urllib.URLopener()
    imgopen.retrieve('https://s3.amazonaws.com/eds-uga-csci8360/data/project3/images/%s.png' % file, "./data/images/%s.png" % file)
