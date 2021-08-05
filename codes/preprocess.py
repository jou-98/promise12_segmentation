import nrrd 
import glob 
from skimage.exposure import equalize_adapthist, equalize_hist
import cv2 
import numpy as np
import matplotlib.pyplot as plt


N_SEGMENTS = 2 # For now
cztz = {'tz':"transitional zone|tz",'cz':"central zone|cz"}
label_re = {'tzcz':"transitional zone|tz|central zone|cz",\
            'pz':"peripheral zone|pz",'b':"bladder",'p':"prostate"}


# Checks if the two strings belong to the same label
def same_label(name1,name2):
    for exp1 in label_re.values():
        for exp2 in label_re.values():
            if re.search(exp1,name1,re.IGNORECASE) and re.search(exp2,name2,re.IGNORECASE):
                return exp1 == exp2

# Flip all transitional zone labels to central zone labels and 
# change tz segment name to None
def merge_cz_tz(arr, header):
    cz_i = -1
    tz_i = -1
    for i in range(N_SEGMENTS):
        segName = header['Segment'+str(i)+'_Name']
        if re.search(cztz['cz'],segName,re.IGNORECASE):
            cz_i = i
        elif re.search(cztz['tz'],segName,re.IGNORECASE):
            tz_i = i
    if cz_i != -1 and tz_i != -1:
        tz_n = int(header['Segment'+str(tz_i)+'_LabelValue'])
        cz_n = int(header['Segment'+str(cz_i)+'_LabelValue'])
        print(f'cz_n={cz_n}')
        arr[arr==tz_n]=cz_n # FLip all tz to cz
        header['Segment'+str(tz_i)+'_Name']='None'
    return arr, header

def equalize_histogram(images,img_rows=480,img_cols=480):
    new_imgs = np.zeros([len(images),img_rows, img_cols])
    for mm, img in enumerate(images):
        img = equalize_adapthist(img,clip_limit=0.05)
        new_imgs[mm] = cv2.resize(img,(img_rows,img_cols),interpolation=cv2.INTER_NEAREST)
    return (new_imgs*255).astype(np.int16)





if __name__=='__main__':  
    arg = '../Dataset/*'
    files = glob.glob(arg)
    seg_shapes = dict()
    img_shapes = dict()
    for file in files:
        if not '.nrrd' in file:
            continue
        if 'seg' in file:
            arr, header = nrrd.read(file)
            arr[arr!=0]=1
            arr = np.moveaxis(arr,-1,0)
            #print(f'Shape of {file[:-9]+".seg.bin"} is {arr.shape}')
            arr.astype(np.float32).tofile(file[:-9]+'.seg.bin')
            seg_shapes[file[-11:-9]] = arr.shape
        else:
            arr, header = nrrd.read(file)
            arr = arr / np.max(arr)
            print(f'Max is {np.max(arr)} and Min is {np.min(arr)}.')
            arr = np.moveaxis(arr,-1,0)
            arr.astype(np.float32).tofile(file[:-5]+'.bin')
            img_shapes[file[-7:-5]] = arr.shape
    print(seg_shapes)
    print(img_shapes)
    for key in seg_shapes:
        if(seg_shapes[key] != img_shapes[key]): print(f'Something is wrong')


