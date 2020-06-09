import cv2
import math
import numpy as np
import scipy as sp
import pandas as pd
import os

train_path = "global-wheat-detection/train.csv"
train_image_path = "global-wheat-detection/train"
output_train_image_path = "global-wheat-detection/output-train"

def read_labelling_csv(train_path):
    train_data = pd.read_csv(train_path)
    return train_data


# # ecrire à la bonne façon
# bboxs=[ bbox[1:-1].split(', ') for bbox in train['bbox']]
# bboxs=[ f"{int(float(bbox[0]))},{int(float(bbox[1]))},{int(float(bbox[0]))+int(float(bbox[2]))},{int(float(bbox[1])) + int(float(bbox[3]))},wheat" for bbox in bboxs]
# train['bbox_']=bboxs
# train.head()
# with open("annotations.csv","w") as file:
#     for idx in range(len(train_df)):
#         file.write(train_img+"/"+train_df.iloc[idx,0]+".jpg"+","+train_df.iloc[idx,1]+"\n")


def draw_bbox(image, bbox):
    for box in bbox:
        rgb = np.floor(np.random.rand(3) * 1024).astype('int')
        image = overlay_box(im=image, box=box, rgb=rgb, stroke=5)
    return image


def overlay_box(im, box, rgb, stroke=1):
    """
    Method to overlay single box on image

    """
    # # --- Convert coordinates to integers
    # box = box[1:-1].split(',')
    # # box = [float(b) for b in box]
    box = [int(float(b)) for b in box]
    
    # --- Extract coordinates
    x1, y1, x2, y2 = box  # xmin, ymin, width, height
    # y2 = y1 + height
    # x2 = x1 + width
    

           
    im[y1:y1 + stroke, x1:x2-1] = rgb
    im[y2-1:y2-1 + stroke, x1:x2-1] = rgb
    im[y1:y2-1, x1:x1 + stroke] = rgb
    im[y1:y2-1, x2-1:x2-1 + stroke] = rgb

    return im


def load_image(image_id, train_image_path):
    file_path = image_id + ".jpg"
    image = cv2.imread(os.path.join(train_image_path, file_path))
    return image # cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


train_data = read_labelling_csv(train_path)
train_data[['x1', 'y1', 'width', 'height']] = train_data['bbox'].str[1:-1].str.split(',', expand=True).astype(float)

train_data['x2'] = train_data.apply(lambda x: x['x1'] + x['width']-1, axis=1)
train_data['y2'] = train_data.apply(lambda x: x['y1'] + x['height']-1, axis=1)

train_data.to_csv("train_full.csv", sep=";")
# train_images = train_data["image_id"][:sample_len].progress_apply(load_image)
for indice, id in enumerate(train_data['image_id'].unique()):
    print(id)
    img = load_image(id, train_image_path)
    bbox = train_data.loc[train_data['image_id'] == id, ['x1', 'y1', 'x2', 'y2']].values
    img = draw_bbox(img, bbox)
    name = f"{indice}_{id}"
    cv2.imwrite(os.path.join(output_train_image_path, name)+".jpg", img)  # f'{indice}"_"{id}'
