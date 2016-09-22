import subprocess

'''
NOTE:
Before running set AWS_ACCESS_ID and AWS_SECRET_KEY environment variables
'''

#settings:
X_train = "./data/X_train_small.txt"
X_test = "./data/X_test_small.txt"


#download files
with open(X_train,'r') as f:
    filenames = f.readlines()

for i in xrange(len(filenames)):
    file = filenames[i].replace('\n','').strip()
    cmd = 'wget --directory-prefix=binaries/train https://s3.amazonaws.com/eds-uga-csci8360/data/project2/binaries/'+file+'.bytes'
    #cmd = 'aws s3 cp s3://eds-uga-csci8360/data/project2/binaries/%s.bytes data/binaries/%s.bytes' % (file,file)
    print 'Getting file %i of %i: %s' % (i+1,len(filenames),file)
    push=subprocess.Popen(cmd, shell=True, stdout = subprocess.PIPE)
    print push.returncode


with open(X_test,'r') as f:
    filenames = f.readlines()

for i in xrange(len(filenames)):
    file = filenames[i].replace('\n','').strip()
    #cmd = 'aws s3 cp s3://eds-uga-csci8360/data/project2/binaries/%s.bytes data/binaries/%s.bytes' % (file,file)
    cmd = 'wget --directory-prefix=binaries/test https://s3.amazonaws.com/eds-uga-csci8360/data/project2/binaries/'+file+'.bytes'
    print 'Getting file %i of %i: %s' % (i+1,len(filenames),file)
    push=subprocess.Popen(cmd, shell=True, stdout = subprocess.PIPE)
    print push.returncode
    