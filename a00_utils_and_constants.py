# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

import numpy as np
import gzip
import pickle
import os
import glob
import time
import cv2
import datetime
import pandas as pd
from collections import Counter, defaultdict
import random
import shutil
import operator
# import pyvips
from PIL import Image
import platform
import json


# Change this to location of Open Images!
# DATASET_PATH = 'E:/Projects_M2/2019_06_Google_Open_Images/input/data_detection/'
DATASET_PATH = './input/'


ROOT_PATH = os.path.dirname(os.path.realpath(__file__)) + '/'
INPUT_PATH = ROOT_PATH + 'input/'
OUTPUT_PATH = ROOT_PATH + 'output/'
if not os.path.isdir(OUTPUT_PATH):
    os.mkdir(OUTPUT_PATH)
MODELS_PATH = ROOT_PATH + 'models/'
if not os.path.isdir(MODELS_PATH):
    os.mkdir(MODELS_PATH)
SUBM_PATH = ROOT_PATH + 'subm/'
if not os.path.isdir(SUBM_PATH):
    os.mkdir(SUBM_PATH)

# https://storage.googleapis.com/openimages/challenge_2018/bbox_labels_500_hierarchy_visualizer/circle.html

LEVEL_1_LABELS = ['Accordion', 'Adhesive tape', 'Airplane', 'Alarm clock', 'Alpaca', 'Ambulance', 'Ant', 'Antelope',
                  'Apple', 'Artichoke', 'Asparagus', 'Backpack', 'Bagel', 'Balloon', 'Banana', 'Barge', 'Barrel',
                  'Baseball bat', 'Baseball glove', 'Bat', 'Bathroom cabinet', 'Bathtub', 'Beaker', 'Bee', 'Beehive',
                  'Beer', 'Bell pepper', 'Belt', 'Bench', 'Bicycle', 'Bicycle helmet', 'Bicycle wheel', 'Bidet',
                  'Billboard', 'Billiard table', 'Binoculars', 'Blender', 'Blue jay', 'Book', 'Bookcase', 'Boot',
                  'Bottle', 'Bow and arrow', 'Bowl', 'Box', 'Boy', 'Brassiere', 'Bread', 'Briefcase', 'Broccoli',
                  'Bronze sculpture', 'Brown bear', 'Bull', 'Burrito', 'Bus', 'Bust', 'Butterfly', 'Cabbage',
                  'Cabinetry', 'Cake', 'Cake stand', 'Camel', 'Camera', 'Canary', 'Candle', 'Candy', 'Cannon',
                  'Canoe', 'Carrot', 'Cart', 'Castle', 'Cat', 'Caterpillar', 'Cattle', 'Ceiling fan', 'Cello',
                  'Centipede', 'Chair', 'Cheetah', 'Chest of drawers', 'Chicken', 'Chopsticks', 'Christmas tree',
                  'Coat', 'Cocktail', 'Coconut', 'Coffee', 'Coffee cup', 'Coffee table', 'Coffeemaker', 'Coin',
                  'Common fig', 'Computer keyboard', 'Computer monitor', 'Computer mouse', 'Convenience store',
                  'Cookie', 'Corded phone', 'Countertop', 'Cowboy hat', 'Crab', 'Cricket ball', 'Crocodile',
                  'Croissant', 'Crown', 'Crutch', 'Cucumber', 'Cupboard', 'Curtain', 'Cutting board', 'Dagger',
                  'Deer', 'Desk', 'Dice', 'Digital clock', 'Dinosaur', 'Dog', 'Dog bed', 'Doll', 'Dolphin',
                  'Door', 'Door handle', 'Doughnut', 'Dragonfly', 'Drawer', 'Dress', 'Drinking straw', 'Drum',
                  'Duck', 'Dumbbell', 'Eagle', 'Earrings', 'Egg', 'Elephant', 'Envelope', 'Falcon', 'Fedora',
                  'Filing cabinet', 'Fire hydrant', 'Fireplace', 'Flag', 'Flashlight', 'Flowerpot', 'Flute',
                  'Food processor', 'Football', 'Football helmet', 'Fork', 'Fountain', 'Fox', 'French fries',
                  'Frog', 'Frying pan', 'Gas stove', 'Giraffe', 'Girl', 'Glasses', 'Goat', 'Goggles', 'Goldfish',
                  'Golf ball', 'Golf cart', 'Gondola', 'Goose', 'Grape', 'Grapefruit', 'Guacamole', 'Guitar',
                  'Hamburger', 'Hamster', 'Handbag', 'Handgun', 'Harbor seal', 'Harp', 'Harpsichord', 'Headphones',
                  'Helicopter', 'High heels', 'Honeycomb', 'Horn', 'Horse', 'Hot dog', 'House', 'Houseplant',
                  'Human arm', 'Human beard', 'Human ear', 'Human eye', 'Human face', 'Human foot', 'Human hair',
                  'Human hand', 'Human head', 'Human leg', 'Human mouth', 'Human nose', 'Ice cream', 'Infant bed',
                  'Jacket', 'Jaguar', 'Jeans', 'Jellyfish', 'Jet ski', 'Jug', 'Juice', 'Kangaroo', 'Kettle',
                  'Kitchen & dining room table', 'Kitchen knife', 'Kite', 'Knife', 'Ladder', 'Ladybug', 'Lamp',
                  'Lantern', 'Laptop', 'Lavender', 'Lemon', 'Leopard', 'Lifejacket', 'Light bulb', 'Light switch',
                  'Lighthouse', 'Lily', 'Limousine', 'Lion', 'Lizard', 'Lobster', 'Loveseat', 'Lynx', 'Man',
                  'Mango', 'Maple', 'Measuring cup', 'Mechanical fan', 'Microphone', 'Microwave oven', 'Miniskirt',
                  'Mirror', 'Missile', 'Mixer', 'Mobile phone', 'Monkey', 'Motorcycle', 'Mouse', 'Muffin', 'Mug',
                  'Mule', 'Mushroom', 'Musical keyboard', 'Nail', 'Necklace', 'Nightstand', 'Oboe', 'Office building',
                  'Orange', 'Organ', 'Ostrich', 'Otter', 'Oven', 'Owl', 'Oyster', 'Paddle', 'Palm tree', 'Pancake',
                  'Paper towel', 'Parachute', 'Parrot', 'Pasta', 'Peach', 'Pear', 'Pen', 'Penguin', 'Piano',
                  'Picnic basket', 'Picture frame', 'Pig', 'Pillow', 'Pineapple', 'Pitcher', 'Pizza', 'Plastic bag',
                  'Plate', 'Platter', 'Polar bear', 'Pomegranate', 'Popcorn', 'Porch', 'Porcupine', 'Poster',
                  'Potato', 'Power plugs and sockets', 'Pressure cooker', 'Pretzel', 'Printer', 'Pumpkin',
                  'Punching bag', 'Rabbit', 'Raccoon', 'Radish', 'Raven', 'Refrigerator', 'Rhinoceros', 'Rifle',
                  'Ring binder', 'Rocket', 'Roller skates', 'Rose', 'Rugby ball', 'Ruler', 'Salad',
                  'Salt and pepper shakers', 'Sandal', 'Saucer', 'Saxophone', 'Scarf', 'Scissors', 'Scoreboard',
                  'Screwdriver', 'Sea lion', 'Sea turtle', 'Seahorse', 'Seat belt', 'Segway', 'Serving tray',
                  'Sewing machine', 'Shark', 'Sheep', 'Shelf', 'Shirt', 'Shorts', 'Shotgun', 'Shower', 'Shrimp',
                  'Sink', 'Skateboard', 'Ski', 'Skull', 'Skyscraper', 'Slow cooker', 'Snail', 'Snake', 'Snowboard',
                  'Snowman', 'Snowmobile', 'Snowplow', 'Sock', 'Sofa bed', 'Sombrero', 'Sparrow', 'Spatula',
                  'Spider', 'Spoon', 'Sports uniform', 'Squirrel', 'Stairs', 'Starfish', 'Stationary bicycle',
                  'Stool', 'Stop sign', 'Strawberry', 'Street light', 'Stretcher', 'Studio couch',
                  'Submarine sandwich', 'Suit', 'Suitcase', 'Sun hat', 'Sunflower', 'Sunglasses', 'Surfboard',
                  'Sushi', 'Swan', 'Swim cap', 'Swimming pool', 'Swimwear', 'Sword', 'Table tennis racket',
                  'Tablet computer', 'Taco', 'Tank', 'Tap', 'Tart', 'Taxi', 'Tea', 'Teapot', 'Teddy bear',
                  'Television', 'Tennis ball', 'Tennis racket', 'Tent', 'Tiara', 'Tick', 'Tie', 'Tiger', 'Tin can',
                  'Tire', 'Toaster', 'Toilet', 'Toilet paper', 'Tomato', 'Torch', 'Tortoise', 'Towel', 'Tower',
                  'Traffic light', 'Train', 'Training bench', 'Treadmill', 'Tripod', 'Trombone', 'Truck',
                  'Trumpet', 'Turkey', 'Umbrella', 'Van', 'Vase', 'Vehicle registration plate', 'Violin',
                  'Volleyball', 'Waffle', 'Wall clock', 'Washing machine', 'Waste container', 'Watch',
                  'Watermelon', 'Whale', 'Wheel', 'Wheelchair', 'Whiteboard', 'Willow', 'Window',
                  'Window blind', 'Wine', 'Wine glass', 'Winter melon', 'Wok', 'Woman', 'Wood-burning stove',
                  'Woodpecker', 'Wrench', 'Zebra', 'Zucchini']


LEVEL_2_LABELS = ['Toy', 'Home appliance', 'Plumbing fixture', 'Office supplies', 'Tableware', 'Kitchen appliance',
                  'Couch', 'Bed', 'Table', 'Clock', 'Sculpture', 'Traffic sign', 'Building', 'Person', 'Dessert',
                  'Fruit', 'Shellfish', 'Squash', 'Sandwich', 'Tree', 'Flower', 'Car', 'Boat', 'Aircraft', 'Hat',
                  'Skirt', 'Glove', 'Trousers', 'Footwear', 'Luggage and bags', 'Helmet', 'Bird',
                  'Marine invertebrates', 'Beetle', 'Moths and butterflies', 'Bear', 'Marine mammal', 'Turtle',
                  'Fish', 'Personal care', 'Musical instrument', 'Ball', 'Racket', 'Weapon', 'Telephone',
                  'Drink']

LEVEL_3_LABELS = ['Seafood', 'Watercraft', 'Insect', 'Carnivore']

# Some classes upper to make more than one class for single net
LEVEL_4_LABELS = ['Vegetable', 'Land vehicle', 'Reptile', 'Invertebrate']

# Some classes upper to make more than one class for single net
LEVEL_5_LABELS = ['Furniture', 'Vehicle', 'Animal']

# Classes with less than 500 samples in train
LEVEL_1_LABELS_LOW_SAMPLES = ['Adhesive tape', 'Alarm clock', 'Ambulance', 'Artichoke', 'Asparagus', 'Bathroom cabinet',
                              'Beaker', 'Belt', 'Bidet', 'Binoculars', 'Blender', 'Blue jay', 'Briefcase', 'Burrito',
                              'Cabbage', 'Cake stand', 'Canary', 'Ceiling fan', 'Centipede', 'Coffeemaker', 'Common fig',
                              'Corded phone', 'Cricket ball', 'Croissant', 'Crutch', 'Cutting board', 'Dagger',
                              'Digital clock', 'Dog bed', 'Drinking straw', 'Dumbbell', 'Envelope', 'Filing cabinet',
                              'Fire hydrant', 'Flashlight', 'Flute', 'Food processor', 'Frying pan', 'Golf ball',
                              'Guacamole', 'Harp', 'Harpsichord', 'Honeycomb', 'Hot dog', 'Infant bed',
                              'Kitchen knife', 'Light switch', 'Limousine', 'Lynx', 'Mango', 'Measuring cup',
                              'Microwave oven', 'Mixer', 'Nail', 'Oboe', 'Organ', 'Paper towel', 'Picnic basket',
                              'Pitcher', 'Popcorn', 'Porcupine', 'Power plugs and sockets', 'Pressure cooker',
                              'Pretzel', 'Printer', 'Punching bag', 'Raccoon', 'Ring binder', 'Rugby ball', 'Ruler',
                              'Salt and pepper shakers', 'Scissors', 'Screwdriver', 'Seahorse', 'Seat belt',
                              'Serving tray', 'Sewing machine', 'Shower', 'Slow cooker', 'Snowmobile', 'Snowplow',
                              'Spatula', 'Stationary bicycle', 'Stop sign', 'Stretcher', 'Submarine sandwich',
                              'Tiara', 'Tick', 'Toaster', 'Toilet paper', 'Torch', 'Towel', 'Training bench',
                              'Treadmill', 'Winter melon', 'Wood-burning stove', 'Wrench']

# All labels
ALL_LABELS = LEVEL_1_LABELS + LEVEL_2_LABELS + LEVEL_3_LABELS + LEVEL_4_LABELS + LEVEL_5_LABELS


def save_in_file(arr, file_name):
    pickle.dump(arr, gzip.open(file_name, 'wb+', compresslevel=3))


def load_from_file(file_name):
    return pickle.load(gzip.open(file_name, 'rb'))


def save_in_file_fast(arr, file_name):
    pickle.dump(arr, open(file_name, 'wb'))


def load_from_file_fast(file_name):
    return pickle.load(open(file_name, 'rb'))


def show_image(im, name='image'):
    cv2.imshow(name, im.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_resized_image(P, w=1000, h=1000):
    res = cv2.resize(P.astype(np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
    show_image(res)


def get_date_string():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")


def sort_dict_by_values(a, reverse=True):
    sorted_x = sorted(a.items(), key=operator.itemgetter(1), reverse=reverse)
    return sorted_x


def value_counts_for_list(lst):
    a = dict(Counter(lst))
    a = sort_dict_by_values(a, True)
    return a


def read_single_image(path):
    use_pyvips = False
    try:
        if not use_pyvips:
            img = np.array(Image.open(path))
        else:
            # Much faster in case you have pyvips installed (uncomment import pyvips in top of file)
            img = pyvips.Image.new_from_file(path, access='sequential')
            img = np.ndarray(buffer=img.write_to_memory(),
                         dtype=np.uint8,
                         shape=[img.height, img.width, img.bands])
    except:
        try:
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        except:
            print('Fail')
            return None

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    if img.shape[2] == 2:
        img = img[:, :, :1]

    if img.shape[2] == 1:
        img = np.concatenate((img, img, img), axis=2)

    if img.shape[2] > 3:
        img = img[:, :, :3]

    return img


def get_description_for_labels():
    # You can find file here:
    # https://raw.githubusercontent.com/StrongRay/YOLOV3-PMD/master/class-descriptions-boxable.csv
    out = open(INPUT_PATH + 'class-descriptions-boxable.csv')
    lines = out.readlines()
    ret_1, ret_2 = dict(), dict()
    for l in lines:
        arr = l.strip().split(',')
        ret_1[arr[0]] = arr[1]
        ret_2[arr[1]] = arr[0]
    return ret_1, ret_2


def read_image_bgr_fast(path):
    img2 = read_single_image(path)
    img2 = img2[:, :, ::-1]
    return img2


def get_subcategories(sub_cat, upper_cat, level, l, d1, sub):
    ret = []
    sub_cat[upper_cat] = ([], [])
    for j, k in enumerate(l[sub]):
        nm = d1[k['LabelName']]
        sub_cat[upper_cat][1].append(nm)
        if nm in sub_cat:
            continue
        ret.append(nm)
        if 'Subcategory' in k:
            get_subcategories(sub_cat, nm, level + 1, l, d1, 'Subcategory')
        else:
            sub_cat[nm] = ([upper_cat], [])
    return ret


def get_hierarchy_structures():
    sub_cat = dict()
    part_cat = dict()
    d1, d2 = get_description_for_labels()
    arr = json.load(open(INPUT_PATH + 'bbox_labels_600_hierarchy.json', 'r'))
    lst = dict(arr.items())['Subcategory']
    for i, l in enumerate(lst):
        nm = d1[l['LabelName']]
        if 'Subcategory' in l:
            get_subcategories(sub_cat, nm, 1, l, d1, 'Subcategory')
        else:
            if nm in sub_cat:
                print('Strange!')
                exit()
            sub_cat[nm] = [], []
    return sub_cat


def set_parents(parents, name_list, l, d1):
    for j, k in enumerate(l['Subcategory']):
        nm = d1[k['LabelName']]
        parents[nm] += name_list
        if 'Subcategory' in k:
            set_parents(parents, name_list + [nm], k, d1)


def get_parents_labels():
    d1, d2 = get_description_for_labels()
    parents = dict()
    for r in d2.keys():
        parents[r] = []

    arr = json.load(open(INPUT_PATH + 'bbox_labels_600_hierarchy.json', 'r'))
    lst = dict(arr.items())['Subcategory']
    for i, l in enumerate(lst):
        nm = d1[l['LabelName']]
        if 'Subcategory' in l:
            set_parents(parents, [nm], l, d1)
    # print(parents)
    for p in parents:
        parents[p] = list(set(parents[p]))
    return parents


def get_description_for_labels_500():
    out = open(INPUT_PATH + 'challenge-2018-class-descriptions-500.csv')
    lines = out.readlines()
    ret_1, ret_2 = dict(), dict()
    for l in lines:
        arr = l.strip().split(',')
        ret_1[arr[0]] = arr[1]
        ret_2[arr[1]] = arr[0]
    return ret_1, ret_2


def random_intensity_change1(img, min_change=-20, max_change=20, separate_channel=True):
    img = img.astype(np.float32)
    delta = random.randint(min_change, max_change)
    for j in range(3):
        if separate_channel:
            delta = random.randint(min_change, max_change)
        img[:, :, j] += delta
    img[img < 0] = 0
    img[img > 255] = 255
    return img.astype(np.uint8)
