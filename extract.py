import os
import json
import random
import numpy as np
from PIL import Image
from glob import glob

# 0 = right eye pupil
# 1 = left eye pupil
# 2 = right mouth corner
# 3 = left mouth corner
# 4 = outer end of right eye brow
# 5 = inner end of right eye brow
# 6 = inner end of left eye brow
# 7 = outer end of left eye brow
# 8 = right temple
# 9 = outer corner of right eye
# 10 = inner corner of right eye
# 11 = inner corner of left eye
# 12 = outer corner of left eye
# 13 = left temple
# 14 = tip of nose
# 15 = right nostril
# 16 = left nostril
# 17 = centre point on outer edge of upper lip
# 18 = centre point on outer edge of lower lip
# 19 = tip of chin

def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

def load_pts(fileName):

    data = []
    start = False

    with open(fileName, 'r') as file:
        for line in file:

            if line.find('{') > -1:
                start = True

            elif line.find('}') > -1:
                break;

            elif start:
                data.append([ int(float(x)) for x in line.strip().split(' ')])

    return data

def extract_features(fileName, **kwargs):

    xOffset = kwargs.get('x_offset', 64)
    yOffset = kwargs.get('y_offset', 15)

    points = load_pts(fileName)

    features = {}

    for id in range(len(points)):

        feature = points[id]

        x = str(feature[0] - xOffset)
        y = str(feature[1] - yOffset)

        if not x in features:
            features[x] = {}

        if not y in features[x]:
            features[x][y] = []

        features[x][y].append(id)

    return features

def extract_blocks_around_pts(fileName, pts, **kwargs):

    xOffset = kwargs.get('x_offset', 64)
    yOffset = kwargs.get('y_offset', 15)
    blockSize = kwargs.get('block_pixels', 64)
    halfBlock = int(blockSize / 2)

    # Load the image
    image = Image.open(fileName)

    # Images have the dimensions:
    # Width: 384 px
    # Height: 286 px

    # Get the pixel data as floating point
    pixels = np.reshape(np.asarray(image.getdata(), dtype=float), (-1, kwargs.get('width', 384))) / kwargs.get('max_alpha', 255.);

    # Slice the top, bottom and sides from the image to create a base2 256x256 image
    pixels = np.around(pixels, kwargs.get('round_to', 4))
    #[yOffset:-yOffset, xOffset:-xOffset]
    blocks = {}

    for ptId in range(len(pts)):
        pt = pts[ptId]
        blocks[str(ptId)] = pixels[(pt[1]-halfBlock):(pt[1]+halfBlock), (pt[0]-halfBlock):(pt[0]+halfBlock)].ravel().tolist()

    return blocks

def extract_blocks(fileName, **kwargs):

    xOffset = kwargs.get('x_offset', 64)
    yOffset = kwargs.get('y_offset', 15)
    blockSize = kwargs.get('block_pixels', 64)

    # Load the image
    image = Image.open(fileName)

    # Images have the dimensions:
    # Width: 384 px
    # Height: 286 px

    # Get the pixel data as floating point
    pixels = np.reshape(np.asarray(image.getdata(), dtype=float), (-1, kwargs.get('width', 384))) / kwargs.get('max_alpha', 255.);

    # Slice the top, bottom and sides from the image to create a base2 256x256 image
    pixels = np.around(pixels[yOffset:-yOffset, xOffset:-xOffset], kwargs.get('round_to', 4))

    # Extract blocks from the image, left to right, top to bottom
    return blockshaped(pixels, blockSize, blockSize)

def extract_face_cascades(face, **kwargs):

    # How many pixels to use for each feature
    blockSize = kwargs.get('block_pixels', 64)

    # Get the image data as a series of blocks, and the features
    blocks = extract_blocks(face['pgm'], **kwargs)
    features = extract_features(face['pts'], **kwargs)

    dataBlocks = {};

    xPos = 0
    yPos = 0
    for block in blocks:

        x = str(xPos)
        y = str(yPos)

        # Create the block, and add the pixel data
        if not x in dataBlocks:
            dataBlocks[x] = {}

        if not y in dataBlocks[x]:
            dataBlocks[x][y] = { 'features': {} }

        dataBlocks[x][y]['data'] = block.ravel().tolist()

        for xIt in range(blockSize):
            for yIt in range(blockSize):

                xPtr = str(xIt + xPos)
                yPtr = str(yIt + yPos)

                if xPtr in features and yPtr in features[xPtr]:

                    for feature in features[xPtr][yPtr]:
                        dataBlocks[x][y]['features'][feature] = [ xPtr, yPtr ]

        # Check to see if we've got a feature in this block

        xPos += blockSize
        if xPos >= 256:
            yPos += blockSize;
            xPos = 0

    return dataBlocks

def extract_snapshots(fileName, **kwargs):
    points = load_pts(fileName['pts'])
    return extract_blocks_around_pts(fileName['pgm'], points, **kwargs)

def get_only_features(fileName, **kwargs):

    featureBlocks = extract_face_cascades(fileName, **kwargs)

    toReturn = []

    for x in featureBlocks:
        for y in featureBlocks[x]:
            if len(featureBlocks[x][y]['features']):
                toReturn.append(featureBlocks[x][y])

    return toReturn

def combine_features(superSet):

    toReturn = {}

    for blockSet in superSet:
        for block in blockSet:
            for feature in block['features']:
                if not feature in toReturn:
                    toReturn[feature] = []

                toReturn[feature].append(block['data'])

    return toReturn

def combine_snapshots(superSet):

    toReturn = {}

    for data in superSet:
        for key, item in data.items():
            if not key in toReturn:
                toReturn[key] = []

            toReturn[key].append(item)

    return toReturn

data = []
COUNTER = 0
files = glob('./face-data/*.pgm')

r = list(range(len(files)))
random.shuffle(r)

for idx in r:

    fileName = files[idx]
    fileName = os.path.splitext(fileName)[0]

    print('Processing', fileName, '...')

    if not os.path.isfile(fileName + '.pts'):
        throw ('Could not find face data for ' + fileName)

    data.append(
        extract_snapshots({

            'pgm': fileName + '.pgm',
            'pts': fileName + '.pts'

        }, round_to=3, block_pixels=32)
    )

    COUNTER += 1
    if COUNTER > 20:
        break

with open('index.js', 'w') as fout:
    prefix = 'export default '
    fout.write(prefix + json.dumps(combine_snapshots(data), sort_keys=True, separators=(',',':')))

print('Feature data dumped to file!')
