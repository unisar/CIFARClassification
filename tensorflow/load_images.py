import numpy as np
from PIL import Image

img_ids = "../data/X_train.txt"
with open(img_ids,'r') as f:
    filenames = f.readlines()

inputs = []
    
input = np.empty((0,3,32,32))
for i in xrange(len(filenames)):
    if i % 1000 == 0:
        inputs.append(input)
        input = np.empty((0,3,32,32))
    print 'Loading file %i of %i:' % (i+1,len(filenames))
    file = filenames[i].replace('\n','').strip()
    image = Image.open("../data/images/%s.png" % file)
    img = np.array(image,dtype='float64')/256
    img = img.transpose(2, 0, 1).reshape(1,3,32,32)
    input = np.concatenate((input,img),axis=0)
    image.close()
inputs.append(input)

final = np.empty((0,3,32,32))
for i in xrange(len(inputs)):
    print 'Combining chunks %i of %i:' % (i+1,len(inputs))
    final = np.concatenate((final,inputs[i]),axis=0)
    
np.save('X_train', final)

inputs = []

input = np.empty((0,3,32,32))
for i in xrange(len(filenames)):
    if i % 1000 == 0:
        inputs.append(input)
        input = np.empty((0,3,32,32))
    print 'Loading file %i of %i:' % (i+1,len(filenames))
    file = filenames[i].replace('\n','').strip()
    image = Image.open("../data/images/%s.png" % file).transpose(Image.FLIP_LEFT_RIGHT)
    img = np.array(image,dtype='float64')/256
    img = img.transpose(2, 0, 1).reshape(1,3,32,32)
    input = np.concatenate((input,img),axis=0)
    image.close()
inputs.append(input)

final = np.empty((0,3,32,32))
for i in xrange(len(inputs)):
    print 'Combining chunks %i of %i:' % (i+1,len(inputs))
    final = np.concatenate((final,inputs[i]),axis=0)
    
np.save('X_train_flipped', final)
