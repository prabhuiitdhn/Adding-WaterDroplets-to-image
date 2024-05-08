
import sys
import os
import numpy as np
import cv2
import math
import pyblur
from PIL import Image
from PIL import ImageEnhance
import random


def CheckCollision(DropList):
    """
    This function handle the collision of the drops
    """

    listFinalDrops = []
    Checked_list = []
    list_len = len(DropList)
    # because latter raindrops in raindrop list should has more colision information
    # so reverse list
    DropList.reverse()
    drop_key = 1
    for drop in DropList:
        # if the drop has not been handle
        if drop.getKey() not in Checked_list:
            # if drop has collision with other drops
            if drop.getIfColli():
                # get collision list
                collision_list = drop.getCollisionList()
                # first get radius and center to decide how  will the collision do
                final_x = drop.getCenters()[0] * drop.getRadius()
                final_y = drop.getCenters()[1] * drop.getRadius()
                tmp_devide = drop.getRadius()
                final_R = drop.getRadius() * drop.getRadius()
                for col_id in collision_list:
                    Checked_list.append(col_id)
                    # list start from 0
                    final_x += DropList[list_len - col_id].getRadius() * DropList[list_len - col_id].getCenters()[0]
                    final_y += DropList[list_len - col_id].getRadius() * DropList[list_len - col_id].getCenters()[1]
                    tmp_devide += DropList[list_len - col_id].getRadius()
                    final_R += DropList[list_len - col_id].getRadius() * DropList[list_len - col_id].getRadius()
                final_x = int(round(final_x / tmp_devide))
                final_y = int(round(final_y / tmp_devide))
                final_R = int(round(math.sqrt(final_R)))
                # rebuild drop after handled the collisions
                newDrop = raindrop(drop_key, (final_x, final_y), final_R)
                drop_key = drop_key + 1
                listFinalDrops.append(newDrop)
            # no collision
            else:
                drop.setKey(drop_key)
                drop_key = drop_key + 1
                listFinalDrops.append(drop)

    return listFinalDrops


class raindrop:
    def __init__(self, key, position, radius):
        self.key = key
        self.center_xy = position
        self.radius = radius
        self.ifcol = False
        self.col_with = []
        self.label_map = np.zeros((radius * 5, radius * 4))
        self.alpha_map = np.zeros((radius * 5, radius * 4))
        self.createDrop()
        self.texture = None

    def getIfColli(self):
        return self.ifcol

    def getCollisionList(self):
        return self.col_with

    def createDrop(self):
        cv2.circle(
            self.label_map, (self.radius * 2, self.radius * 3), self.radius, 128, -1
        )

        cv2.ellipse(self.label_map, (self.radius * 2, self.radius * 3),
                    (self.radius, int(1.3 * math.sqrt(3) * self.radius)),
                    0, 180, 360, 128, -1)
        # set alpha map for png
        self.alpha_map = pyblur.GaussianBlur(Image.fromarray(np.uint8(self.label_map)), 10)
        self.alpha_map = np.asarray(self.alpha_map).astype(np.float)
        self.alpha_map = self.alpha_map / np.max(self.alpha_map) * 255.0
        # set label map
        self.label_map[self.label_map > 0] = 1

    def updateTexture(self, bg):
        fg = pyblur.GaussianBlur(Image.fromarray(np.uint8(bg)), 5)

        # fg = pyblur.BoxBlur(Image.fromarray(np.uint8(bg)), dim = 5)

        fg = np.asarray(fg)
        # add fish eye effect to simulate the background
        K = np.array([[30 * self.radius, 0, 2 * self.radius],
                      [0., 20 * self.radius, 3 * self.radius],
                      [0., 0., 1]])
        D = np.array([0.0, 0.0, 0.0, 0.0])
        Knew = K.copy()
        Knew[(0, 1), (0, 1)] = math.pow(self.radius, 1 / 3) * 2 * Knew[(0, 1), (0, 1)]
        fisheye = cv2.fisheye.undistortImage(fg, K, D=D, Knew=Knew)

        tmp = np.expand_dims(self.alpha_map, axis=-1)
        tmp = np.concatenate((fisheye, tmp), axis=2)

        self.texture = Image.fromarray(tmp.astype('uint8'), 'RGBA')
        # most background in drop is flipped
        self.texture = self.texture.transpose(Image.FLIP_TOP_BOTTOM)

    def setCollision(self, col, col_with):
        self.ifcol = col
        self.col_with = col_with

    def setKey(self, key):
        self.key = key

    def getCenters(self):
        return self.center_xy

    def getLabelMap(self):
        return self.label_map

    def getRadius(self):
        return self.radius

    def getKey(self):
        return self.key

    def getAlphaMap(self):
        return self.alpha_map

    def getTexture(self):
        return self.texture



path = r"D:\Camera Blockage\implementation\test_image"
total_number_of_droplets = 30  # total number of droplets we can add
edge_ratio = 0.25  # edge ratio for droplets
raindrop_min_radius = 30
raindrop_max_radius = 50

# Total images to be modified and added raindrop
images = os.listdir(path)

for image in images:
    input_image = os.path.join(path, image)
    PIL_bg_img = Image.open(input_image)
    bg_img = np.asarray(PIL_bg_img)
    label_map = np.zeros_like(bg_img)[:, :, 0]
    imgh, imgw, _ = bg_img.shape  # image height and width
    # random position for the droplets
    ran_pos = [(int(random.random() * imgw), int(random.random() * imgh)) for _ in range(total_number_of_droplets)]

    # list which contains total droplets property
    listRainDrops = []
    # objects of rainDrop class
    rainDrop = raindrop

    for i in range(total_number_of_droplets):
        radius = random.randint(raindrop_min_radius, raindrop_max_radius)
        position = ran_pos[i]
        drop = rainDrop(i + 1, position, radius)
        listRainDrops.append(drop)

    collisionNum = len(listRainDrops)
    listFinalDrops = list(listRainDrops)
    loop = 0

    # Checking collision between two drops. Until collision is zero. The droplets have to be merged.
    while collisionNum > 0:
        loop = loop + 1
        listFinalDrops = list(listFinalDrops)
        collisionNum = len(listFinalDrops)
        label_map = np.zeros_like(label_map)
        # Check Collision
        for drop in listFinalDrops:
            # check the bounding
            (ix, iy) = drop.getCenters()
            radius = drop.getRadius()
            ROI_WL = 2 * radius
            ROI_WR = 2 * radius
            ROI_HU = 3 * radius
            ROI_HD = 2 * radius
            if (iy - 3 * radius) < 0:
                ROI_HU = iy
            if (iy + 2 * radius) > imgh:
                ROI_HD = imgh - iy
            if (ix - 2 * radius) < 0:
                ROI_WL = ix
            if (ix + 2 * radius) > imgw:
                ROI_WR = imgw - ix

            # apply raindrop label map to Image's label map
            drop_label = drop.getLabelMap()

            # check if center has already has drops
            if (label_map[iy, ix] > 0):
                col_ids = np.unique(label_map[iy - ROI_HU:iy + ROI_HD, ix - ROI_WL: ix + ROI_WR])
                col_ids = col_ids[col_ids != 0]
                drop.setCollision(True, col_ids)
                label_map[iy - ROI_HU:iy + ROI_HD, ix - ROI_WL: ix + ROI_WR] = drop_label[
                                                                               3 * radius - ROI_HU:3 * radius + ROI_HD,
                                                                               2 * radius - ROI_WL: 2 * radius + ROI_WR] * drop.getKey()
            else:
                label_map[iy - ROI_HU:iy + ROI_HD, ix - ROI_WL: ix + ROI_WR] = drop_label[
                                                                               3 * radius - ROI_HU:3 * radius + ROI_HD,
                                                                               2 * radius - ROI_WL: 2 * radius + ROI_WR] * drop.getKey()
                # no collision
                collisionNum = collisionNum - 1

        if collisionNum > 0:
            listFinalDrops = CheckCollision(listFinalDrops)

        # generarating binary mask for droplets.
        # 0- droplet reason
        # remaining 1
        alpha_map = np.zeros_like(label_map).astype(np.float64)
        for drop in listFinalDrops:
            (ix, iy) = drop.getCenters()
            radius = drop.getRadius()
            ROI_WL = 2 * radius
            ROI_WR = 2 * radius
            ROI_HU = 3 * radius
            ROI_HD = 2 * radius
            if (iy - 3 * radius) < 0:
                ROI_HU = iy
            if (iy + 2 * radius) > imgh:
                ROI_HD = imgh - iy
            if (ix - 2 * radius) < 0:
                ROI_WL = ix
            if (ix + 2 * radius) > imgw:
                ROI_WR = imgw - ix

            drop_alpha = drop.getAlphaMap()
            alpha_map[iy - ROI_HU:iy + ROI_HD, ix - ROI_WL: ix + ROI_WR] += drop_alpha[
                                                                            3 * radius - ROI_HU:3 * radius + ROI_HD,
                                                                            2 * radius - ROI_WL: 2 * radius + ROI_WR]

        alpha_map = alpha_map / np.max(alpha_map) * 255.0
        # cv2.imshow("test.bmp", alpha_map)

        save_bmp_path = r"D:\Camera Blockage\implementation\test_output\output_bmp"
        file_name = os.path.join(save_bmp_path, image.split('.')[0] + '_before_brightness.bmp')
        cv2.imwrite(file_name, alpha_map)
        # sys.exit()

        # Plotting in the image
        for drop in listFinalDrops:
            (ix, iy) = drop.getCenters()
            radius = drop.getRadius()
            ROIU = iy - 3 * radius
            ROID = iy + 2 * radius
            ROIL = ix - 2 * radius
            ROIR = ix + 2 * radius
            if (iy - 3 * radius) < 0:
                ROIU = 0
                ROID = 5 * radius
            if (iy + 2 * radius) > imgh:
                ROIU = imgh - 5 * radius
                ROID = imgh
            if (ix - 2 * radius) < 0:
                ROIL = 0
                ROIR = 4 * radius
            if (ix + 2 * radius) > imgw:
                ROIL = imgw - 4 * radius
                ROIR = imgw

            tmp_bg = bg_img[ROIU:ROID, ROIL:ROIR, :]
            drop.updateTexture(tmp_bg)
            tmp_alpha_map = alpha_map[ROIU:ROID, ROIL:ROIR]
            output = drop.getTexture()

            tmp_output = np.asarray(output).astype(np.float)[:, :, -1]
            tmp_alpha_map = tmp_alpha_map * (tmp_output / 255)
            tmp_alpha_map = Image.fromarray(tmp_alpha_map.astype('uint8'))

            save_bmp_path = r"D:\Camera Blockage\implementation\test_output\output_bmp"
            tmp_alpha_map.save(os.path.join(save_bmp_path, image.split('.')[0] + '.bmp'))

            edge = ImageEnhance.Brightness(output)
            edge = edge.enhance(edge_ratio)

            PIL_bg_img.paste(edge, (ix - 2 * radius, iy - 3 * radius), tmp_alpha_map)
            PIL_bg_img.paste(output, (ix - 2 * radius, iy - 3 * radius), output)

            output_label = np.array(alpha_map)
            output_label.flags.writeable = True
            output_label[output_label > 0] = 1
            output_label = Image.fromarray(output_label.astype('uint8'))

            save_path = r"D:\Camera Blockage\implementation\test_output\output_image"
            PIL_bg_img.save(os.path.join(save_path, image))

            save_label_path =  r"D:\Camera Blockage\implementation\test_output\output_label"
            cv2.imwrite(os.path.join(save_label_path, image), np.array(output_label) * 255)
    print(str(image) + " : Checked.")

print ("Done.")
sys.exit()