import json
from PIL import Image

def writeImage(name, data):

    width = int(pow(len(item), 0.5))

    # PIL accesses images in Cartesian co-ordinates, so it is Image[columns, rows]
    img = Image.new( 'RGB', (width,width), "white") # create a new black image
    pixels = img.load() # create the pixel map

    for y in range(0, width):
        for x in range(0, width):
            color = int(data[y*width + x] * 255)
            pixels[x,y] = (color, color, color)

    img.save('features/' + name + '.png')
    print('Saved: features/' + name + '.png')

with open('index.js', 'r') as fin:

    data = fin.read();
    data = data[data.find('{'):]
    data = json.loads(data)

    for typeName, values in data.items():
        id=0
        for item in values:
            id+=1
            writeImage(typeName + '_' + str(id), item)
