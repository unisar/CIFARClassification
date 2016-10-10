import subprocess
import urllib

'''
download train and test images from S3 to local drive
'''

#settings:
files = ["./data/X_train.txt","./data/X_test.txt"]

#download files
filenames = []
for file in files:
    with open(file,'r') as f:
        filenames += f.readlines()

for i in xrange(len(filenames)):
    file = filenames[i].replace('\n','').strip()
    print 'Getting file %i of %i: %s' % (i+1,len(filenames),file)    
    cmd = 'aws s3 cp s3://eds-uga-csci8360/data/project3/images/%s.png ./data/images/%s.png' % (file,file)
    push=subprocess.Popen(cmd, shell=True, stdout = subprocess.PIPE)
    print push.returncode