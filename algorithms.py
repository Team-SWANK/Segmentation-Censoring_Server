import scipy.ndimage
import numpy as np
from PIL import Image
import piexif
import io
from skimage import feature

def guassian_blur(img, img_mask, sigma=3):
    img_mask = np.where(img_mask[:,:,:1] > 250, 1, 0)
    mask = img_mask.astype(np.float)
    filter = scipy.ndimage.filters.gaussian_filter(img*mask, sigma=(sigma, sigma, 0))
    weights = scipy.ndimage.filters.gaussian_filter(mask, sigma=(sigma, sigma, 0))
    filter /= weights + 0.001

    filter = filter.astype(np.uint8)
    inv_mask = (mask < 1.0)
    filter -= filter*inv_mask
    img = (img*inv_mask)
    img += filter
    img = img.astype(np.uint8)
    return img

def pixelization(img, mask_img):
    mask_img = np.where(mask_img[:,:,:1] > 250, 1, 0)
    dim_x, dim_y = img.shape[0], img.shape[1]
    img = Image.fromarray(img)
    inv_mask = (mask_img < 1.0)
    imgSmall = img.resize((dim_x//32,dim_y//32),resample=Image.BILINEAR)
    imgSmall = imgSmall.resize(img.size,Image.NEAREST)
    imgSmall -= imgSmall*inv_mask
    img = (img*inv_mask)
    img += imgSmall
    img = img.astype(np.uint8)
    return np.array(img)

def pixel_sort(img, img_mask):
    resized = False
    img_mask = img_mask.reshape(img_mask.shape[0], img_mask.shape[1])
    # resizes mask and changes resized boolean to  True
    if img_mask.shape[0] > 1200 or img_mask.shape[1] > 1200:
        resize_mask = Image.fromarray(img_mask)
        scale_ratio = min(1200/img_mask.shape[0], 1200/img_mask.shape[1])
        resize_mask = resize_mask.resize(tuple([int(x * scale_ratio) for x in resize_mask.size]))
        resize_image = Image.fromarray(img).resize(tuple([int(x * scale_ratio) for x in [img.shape[1], img.shape[0]]]))
        img_mask = np.array(resize_mask)
        resize_image = np.array(resize_image)
        resized = True
    img_mask = np.where(img_mask > 250, 255, 0)
    # Stores beginning and end of row
    selected_row = [-1,-1]
    # Sort pixels horizontally
    for row in range(len(img_mask)):
        for col in range(len(img_mask[row])):
            val = img_mask[row][col]
            if val == 255:
                if selected_row[0] == -1:
                    selected_row[0] = col
            else:
                if selected_row[0] != -1:
                    selected_row[1] = col
                    if resized:
                        np.random.shuffle(resize_image[row][selected_row[0]:selected_row[1]])
                    else:
                        np.random.shuffle(img[row][selected_row[0]:selected_row[1]])
                    selected_row = [-1, -1]
        selected_row = [-1, -1]

    if resized:
        img = np.array(Image.fromarray(resize_image).resize((img.shape[1], img.shape[0])))
    return img

def fill_in(img, img_mask):
    mask_copy = np.where(img_mask[:,:,:1] > 10, 255, 0)
    img_mask = img_mask.reshape(img_mask.shape[0], img_mask.shape[1])# flattening mask to 2D array with singletons
    resized = False# holds whether image was resized or not as those require different implementations

    # resizes mask and changes resized boolean to  True
    if img_mask.shape[0] > 500 or img_mask.shape[1] > 500:
        resize_mask = Image.fromarray(img_mask)
        scale_ratio = min(500/img_mask.shape[0], 500/img_mask.shape[1])
        resize_mask = resize_mask.resize(tuple([int(x * scale_ratio) for x in resize_mask.size]))
        img_mask = np.array(resize_mask)
        resized = True

    # edge detection on image
    edges = feature.canny(img_mask, sigma=3)
    img_mask = np.where(edges == True, 255, 0)

    # RUN DFS
    count = 2
    for row in range(len(img_mask)):
        for col in range(len(img_mask[row])):
            if img_mask[row][col] == 255:
                black_bar_dfs(col, row, img_mask, count)
                count += 1
    xRatio = img.shape[0] / img_mask.shape[0]
    yRatio = img.shape[1] / img_mask.shape[1]
    # rectangular bounding boxes are found on downscaled mask and then the mask with bounding boxes is upscaled
    # into a new mask that is just pasted onto the image
    if(resized):
        for i in range(1, count):
            # mask = np.full_like(img_mask, 0).astype(np.uint8)
            segment = [img_mask.shape[0], img_mask.shape[1], 0 ,0]
            for row in range(len(img_mask)):
                for col in range(len(img_mask[row])):
                    if i == img_mask[row][col]:
                        # left
                        if col < segment[0]:
                            segment[0] = int(col)
                        # top
                        if row < segment[1]:
                            segment[1] = int(row)
                        # right
                        if col > segment[2]:
                            segment[2] = int(col)
                        # bottom
                        if row > segment[3]:
                            segment[3] = int(row)

            locs = np.where(mask_copy[round(segment[1]*yRatio):round(segment[3]*yRatio), round(segment[0]*xRatio):round(segment[2]*xRatio)] == 255)
            if len(locs[0]) < 1:
                continue
            pixels = img[round(segment[1]*yRatio):round(segment[3]*yRatio), round(segment[0]*xRatio):round(segment[2]*xRatio)]
            crop = np.where(mask_copy[round(segment[1]*yRatio):round(segment[3]*yRatio), round(segment[0]*xRatio):round(segment[2]*xRatio)] == 255, 
                np.mean(pixels, axis=(0,1)), 
                img[round(segment[1]*yRatio):round(segment[3]*yRatio), round(segment[0]*xRatio):round(segment[2]*xRatio)])
            img[round(segment[1]*yRatio):round(segment[3]*yRatio), round(segment[0]*xRatio):round(segment[2]*xRatio)] = crop
    # original bounding box for images that weren't resized
    else:
        for i in range(1, count):
            segment = [img.shape[0], img.shape[1], 0 ,0]
            for row in range(len(img_mask)):
                for col in range(len(img_mask[row])):
                    if i == img_mask[row][col]:
                        # left
                        if col < segment[0]:
                            segment[0] = int(col)
                        # top
                        if row < segment[1]:
                            segment[1] = int(row)
                        # right
                        if col > segment[2]:
                            segment[2] = int(col)
                        # bottom
                        if row > segment[3]:
                            segment[3] = int(row)
            locs = np.where(mask_copy[segment[1]:segment[3], segment[0]:segment[2]] == 255)
            if len(locs[0]) < 1:
                continue
            pixels = img[segment[1]:segment[3], segment[0]:segment[2]]
            crop = np.where(mask_copy[segment[1]:segment[3], segment[0]:segment[2]] == 255, 
                np.mean(pixels), 
                img[segment[1]:segment[3], segment[0]:segment[2]])
            img[segment[1]:segment[3], segment[0]:segment[2]] = crop
    return img

def black_bar(img, img_mask):
    BLACK_COLOR = (0,0,0)
    img_mask = img_mask.reshape(img_mask.shape[0], img_mask.shape[1])# flattening mask to 2D array with singletons
    resized = False# holds whether image was resized or not as those require different implementations
    # resizes mask and changes resized boolean to  True
    if img_mask.shape[0] > 500 or img_mask.shape[1] > 500:
        resize_mask = Image.fromarray(img_mask)
        scale_ratio = min(500/img_mask.shape[0], 500/img_mask.shape[1])
        resize_mask = resize_mask.resize(tuple([int(x * scale_ratio) for x in resize_mask.size]))
        img_mask = np.array(resize_mask)
        resized = True
    # edge detection on image 
    edges = feature.canny(img_mask, sigma=3)
    img_mask = np.where(edges == True, 255, 0)
    # RUN DFS
    count = 2
    for row in range(len(img_mask)):
        for col in range(len(img_mask[row])):
            if img_mask[row][col] == 255:
                black_bar_dfs(col, row, img_mask, count)
                count += 1

    # rectangular bounding boxes are found on downscaled mask and then the image with bounding boxes is upscaled 
    # into a new mask that is just pasted onto the image
    if(resized):
        mask = np.full_like(img_mask, 0).astype(np.uint8)
        for i in range(1, count):
            segment = [img_mask.shape[0], img_mask.shape[1], 0 ,0]
            for row in range(len(img_mask)):
                for col in range(len(img_mask[row])):
                    if i == img_mask[row][col]:
                        # left
                        if col < segment[0]:
                            segment[0] = int(col)
                        # top
                        if row < segment[1]:
                            segment[1] = int(row)
                        # right
                        if col > segment[2]:
                            segment[2] = int(col)
                        # bottom
                        if row > segment[3]:
                            segment[3] = int(row)
            mask[segment[1]:segment[3], segment[0]:segment[2]] = 255
        # resize new bounding box mask into original image size
        mask.reshape(mask.shape[0], mask.shape[1])
        resize_mask = Image.fromarray(mask)
        resize_mask = resize_mask.resize(img.size)
        mask = np.array(resize_mask)
        mask = Image.fromarray(np.where(mask > 10, 255, 0).astype(np.uint8))
        img.paste(BLACK_COLOR, mask=mask)
    # original bounding box for images that weren't resized
    else:
        for i in range(1, count):
            segment = [img.size[0], img.size[1], 0 ,0]
            for row in range(len(img_mask)):
                for col in range(len(img_mask[row])):
                    if i == img_mask[row][col]:
                        # left
                        if col < segment[0]:
                            segment[0] = int(col)
                        # top
                        if row < segment[1]:
                            segment[1] = int(row)
                        # right
                        if col > segment[2]:
                            segment[2] = int(col)
                        # bottom
                        if row > segment[3]:
                            segment[3] = int(row)
            img.paste( BLACK_COLOR, [segment[0],segment[1],segment[2],segment[3]])
    # must return as numpy array for postprocessing
    return np.array(img)

def black_bar_dfs(col, row, img_mask, count):
    if row < 0 or row > len(img_mask)-1 or col < 0 or col > len(img_mask[0])-1 or img_mask[row][col] != 255:
        return
    img_mask[row][col] = count

    black_bar_dfs(col-1, row, img_mask, count)#left 
    black_bar_dfs(col, row-1, img_mask, count)#top 
    black_bar_dfs(col+1, row, img_mask, count)#right  
    black_bar_dfs(col, row+1, img_mask, count)#bottom   
    black_bar_dfs(col-1, row-1, img_mask, count)#topleft    
    black_bar_dfs(col+1, row-1, img_mask, count)#topright   
    black_bar_dfs(col-1, row+1, img_mask, count)#bottomleft  
    black_bar_dfs(col+1, row+1, img_mask, count)#bottomright

def adjust_exif2(tags_chosen,exif):
    new_exif = dict(exif)
    tag_space_list = ["0th", "Exif", "GPS", "1st"]
    i =0
    for index,tag_space in enumerate(tag_space_list):
        # print("Tag Space: ",tag_space)
        for chosen in tags_chosen:
            try:
                if index == 0:
                    new_exif[tag_space][piexif.ImageIFD.__getattribute__(piexif.ImageIFD,chosen)] = ""
                elif index == 1:
                    tagType = type(new_exif[tag_space][piexif.ExifIFD.__getattribute__(piexif.ExifIFD,chosen)])
                    if tagType is tuple:
                        new_exif[tag_space][piexif.ExifIFD.__getattribute__(piexif.ExifIFD,chosen)] = (0,0)
                    else:
                        new_exif[tag_space][piexif.ExifIFD.__getattribute__(piexif.ExifIFD,chosen)] = b''
                elif index == 2:
                    tagType = type(new_exif[tag_space][piexif.GPSIFD.__getattribute__(piexif.GPSIFD,chosen)])
                    if tagType is bytes:
                        new_exif[tag_space][piexif.GPSIFD.__getattribute__(piexif.GPSIFD,chosen)] = b''
                    elif tagType is int:
                        new_exif[tag_space][piexif.GPSIFD.__getattribute__(piexif.GPSIFD,chosen)] = 0
                    elif tagType is tuple:
                        new_exif[tag_space][piexif.GPSIFD.__getattribute__(piexif.GPSIFD,chosen)] = (0,0)
                else:
                   new_exif[tag_space][piexif.InteropIFD.__getattribute__(piexif.InteropIFD, chosen)] = ""
                i+=1
                # print("removed: ",chosen, i,"/",len(tags_chosen))
            except: continue #this accounts for the fact that each tag doesnt exist in every tag_space
    try:
        del new_exif['Exif'][piexif.ExifIFD.SceneType]
        # if "ExposureTime" in tags_chosen:
        #     del new_exif['Exif'][piexif.ExifIFD.ExposureTime]
    except:
        pass

    return new_exif

def metadata_erase(img, exif, tags):
    exif = piexif.load(exif)
    new_exif = adjust_exif2(tags,exif)
    new_bytes = piexif.dump(new_exif)
    outputImage = io.BytesIO()
    piexif.insert(new_bytes, img, outputImage)
    return outputImage