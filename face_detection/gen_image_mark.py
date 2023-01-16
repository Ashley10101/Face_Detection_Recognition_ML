from PIL import Image

with open('../face_detection/image_mark.txt') as f:
    content = f.readlines()

outfolder = '../face_detection/adaboost_image'
for i in range(len(content)):
    line = content[i]
    address = line.split(' ')[0]
    name = address.split('/')[-1]
    outpath_face = outfolder + "/" + "1" + name
    outpath_other_0 = outfolder + "/" + "0" + "0" + name
    outpath_other_1 = outfolder + "/" + "0" + "1" + name
    x = int(line.split(' ')[1])
    y = int(line.split(' ')[2])
    w = int(line.split(' ')[3])
    h = int(line.split(' ')[4])
    img = Image.open(address)

    # Intercepting areas that contain faces
    face = img.crop((x,y,x+w,y+h))
    face = face.resize((24,24))
    img = Image.open(address)
    # Intercepting other areas that do not contain faces
    other_0 = img.crop((500,700,600,800))
    other_0 = other_0.resize((24,24))
    other_1 = img.crop((0,700,100,800))
    other_1 = other_0.resize((24,24))
    face.save(outpath_face)
    other_0.save(outpath_other_0)
    other_1.save(outpath_other_1)

    print(i)

    
