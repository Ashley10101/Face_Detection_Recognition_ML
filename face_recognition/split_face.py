from PIL import Image


with open('../face_detection/image_mark.txt') as f:
    content = f.readlines()

outfolder = 'face'
for i in range(len(content)):
    line = content[i]
    address = line.split(' ')[0]
    name = address.split('/')[-1]
    outpath_face = outfolder + "/" + name
    x = int(line.split(' ')[1])
    y = int(line.split(' ')[2])
    w = int(line.split(' ')[3])
    h = int(line.split(' ')[4])
    img = Image.open(address)
    face = img.crop((x,y,x+w,y+h))
    face = face.resize((200,200))
    img = Image.open(address)
    face.save(outpath_face)
    
    print(i)

    
