import numpy as np
from PIL import Image

img_ids = "../data/X_small_test.txt"
with open(img_ids,'r') as f:
    filenames = f.readlines()

#filenames = filenames[0:2]
print filenames
inputs = []
    
input = np.empty((0,32,32,3))
'''
print "Shape: "
print input.shape
print filenames
print xrange(len(filenames))
'''
#Create 10 partitions of all the files
for i in xrange(len(filenames)):
    if i % 10 == 0:
        if i>0:
            inputs.append(input)
            input = np.empty((0,32,32,3))

    print 'Loading file %i of %i:' % (i+1,len(filenames))
    file = filenames[i].replace('\n','').strip()
    image = Image.open("../data/images/%s.png" % file)
    img = np.array(image,dtype='float64')/256
    #print img
    #print img.shape
    #[width,height,depth] => [width,height,depth]
    img = img.reshape(1,32,32,3)
    input = np.concatenate((input,img),axis=0)
    image.close()
inputs.append(input)

#Combine all the np_arrays into one big object
final = np.empty((0,32,32,3))
for i in xrange(len(inputs)):
    print 'Combining chunks %i of %i:' % (i+1,len(inputs))
    final = np.concatenate((final,inputs[i]),axis=0)
 
np.save('./X_small_test_1', final)
