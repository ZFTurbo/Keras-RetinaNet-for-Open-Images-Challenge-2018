# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'


if __name__ == '__main__':
    import os
    gpu_use = 0
    print('GPU use: {}'.format(gpu_use))
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)


from a00_utils_and_constants import *


TENSORFLOW_CATEGORIES = {1: {'name': 'Person', 'id': 1}, 2: {'name': 'Clothing', 'id': 2}, 3: {'name': 'Man', 'id': 3}, 4: {'name': 'Face', 'id': 4}, 5: {'name': 'Tree', 'id': 5}, 6: {'name': 'Plant', 'id': 6}, 7: {'name': 'Woman', 'id': 7}, 8: {'name': 'Vehicle', 'id': 8}, 9: {'name': 'Building', 'id': 9}, 10: {'name': 'Land vehicle', 'id': 10}, 11: {'name': 'Footwear', 'id': 11}, 12: {'name': 'Girl', 'id': 12}, 13: {'name': 'Animal', 'id': 13}, 14: {'name': 'Car', 'id': 14}, 15: {'name': 'Food', 'id': 15}, 16: {'name': 'Wheel', 'id': 16}, 17: {'name': 'Flower', 'id': 17}, 18: {'name': 'Furniture', 'id': 18}, 19: {'name': 'Window', 'id': 19}, 20: {'name': 'House', 'id': 20}, 21: {'name': 'Boy', 'id': 21}, 22: {'name': 'Fashion accessory', 'id': 22}, 23: {'name': 'Table', 'id': 23}, 24: {'name': 'Glasses', 'id': 24}, 25: {'name': 'Suit', 'id': 25}, 26: {'name': 'Auto part', 'id': 26}, 27: {'name': 'Bird', 'id': 27}, 28: {'name': 'Sports equipment', 'id': 28}, 29: {'name': 'Dress', 'id': 29}, 30: {'name': 'Dog', 'id': 30}, 31: {'name': 'Carnivore', 'id': 31}, 32: {'name': 'Human body', 'id': 32}, 33: {'name': 'Jeans', 'id': 33}, 34: {'name': 'Musical instrument', 'id': 34}, 35: {'name': 'Drink', 'id': 35}, 36: {'name': 'Boat', 'id': 36}, 37: {'name': 'Hair', 'id': 37}, 38: {'name': 'Tire', 'id': 38}, 39: {'name': 'Head', 'id': 39}, 40: {'name': 'Cat', 'id': 40}, 41: {'name': 'Watercraft', 'id': 41}, 42: {'name': 'Chair', 'id': 42}, 43: {'name': 'Bike', 'id': 43}, 44: {'name': 'Tower', 'id': 44}, 45: {'name': 'Mammal', 'id': 45}, 46: {'name': 'Skyscraper', 'id': 46}, 47: {'name': 'Arm', 'id': 47}, 48: {'name': 'Toy', 'id': 48}, 49: {'name': 'Sculpture', 'id': 49}, 50: {'name': 'Invertebrate', 'id': 50}, 51: {'name': 'Microphone', 'id': 51}, 52: {'name': 'Poster', 'id': 52}, 53: {'name': 'Insect', 'id': 53}, 54: {'name': 'Guitar', 'id': 54}, 55: {'name': 'Nose', 'id': 55}, 56: {'name': 'Hat', 'id': 56}, 57: {'name': 'Tableware', 'id': 57}, 58: {'name': 'Door', 'id': 58}, 59: {'name': 'Bicycle wheel', 'id': 59}, 60: {'name': 'Sunglasses', 'id': 60}, 61: {'name': 'Baked goods', 'id': 61}, 62: {'name': 'Eye', 'id': 62}, 63: {'name': 'Dessert', 'id': 63}, 64: {'name': 'Mouth', 'id': 64}, 65: {'name': 'Aircraft', 'id': 65}, 66: {'name': 'Airplane', 'id': 66}, 67: {'name': 'Train', 'id': 67}, 68: {'name': 'Jacket', 'id': 68}, 69: {'name': 'Street light', 'id': 69}, 70: {'name': 'Hand', 'id': 70}, 71: {'name': 'Snack', 'id': 71}, 72: {'name': 'Helmet', 'id': 72}, 73: {'name': 'Trousers', 'id': 73}, 74: {'name': 'Bottle', 'id': 74}, 75: {'name': 'Houseplant', 'id': 75}, 76: {'name': 'Horse', 'id': 76}, 77: {'name': 'Desk', 'id': 77}, 78: {'name': 'Palm tree', 'id': 78}, 79: {'name': 'Vegetable', 'id': 79}, 80: {'name': 'Fruit', 'id': 80}, 81: {'name': 'Leg', 'id': 81}, 82: {'name': 'Book', 'id': 82}, 83: {'name': 'Fast food', 'id': 83}, 84: {'name': 'Beer', 'id': 84}, 85: {'name': 'Flag', 'id': 85}, 86: {'name': 'Drum', 'id': 86}, 87: {'name': 'Bus', 'id': 87}, 88: {'name': 'Truck', 'id': 88}, 89: {'name': 'Ball', 'id': 89}, 90: {'name': 'Tie', 'id': 90}, 91: {'name': 'Flowerpot', 'id': 91}, 92: {'name': 'Goggles', 'id': 92}, 93: {'name': 'Motorcycle', 'id': 93}, 94: {'name': 'Picture frame', 'id': 94}, 95: {'name': 'Shorts', 'id': 95}, 96: {'name': 'Sports uniform', 'id': 96}, 97: {'name': 'Moths and butterflies', 'id': 97}, 98: {'name': 'Shelf', 'id': 98}, 99: {'name': 'Shirt', 'id': 99}, 100: {'name': 'Fish', 'id': 100}, 101: {'name': 'Rose', 'id': 101}, 102: {'name': 'Licence plate', 'id': 102}, 103: {'name': 'Couch', 'id': 103}, 104: {'name': 'Weapon', 'id': 104}, 105: {'name': 'Laptop', 'id': 105}, 106: {'name': 'Wine glass', 'id': 106}, 107: {'name': 'Van', 'id': 107}, 108: {'name': 'Wine', 'id': 108}, 109: {'name': 'Duck', 'id': 109}, 110: {'name': 'Bicycle helmet', 'id': 110}, 111: {'name': 'Butterfly', 'id': 111}, 112: {'name': 'Swimming pool', 'id': 112}, 113: {'name': 'Ear', 'id': 113}, 114: {'name': 'Office', 'id': 114}, 115: {'name': 'Camera', 'id': 115}, 116: {'name': 'Stairs', 'id': 116}, 117: {'name': 'Reptile', 'id': 117}, 118: {'name': 'Football', 'id': 118}, 119: {'name': 'Cake', 'id': 119}, 120: {'name': 'Mobile phone', 'id': 120}, 121: {'name': 'Sun hat', 'id': 121}, 122: {'name': 'Coffee cup', 'id': 122}, 123: {'name': 'Christmas tree', 'id': 123}, 124: {'name': 'Computer monitor', 'id': 124}, 125: {'name': 'Helicopter', 'id': 125}, 126: {'name': 'Bench', 'id': 126}, 127: {'name': 'Castle', 'id': 127}, 128: {'name': 'Coat', 'id': 128}, 129: {'name': 'Porch', 'id': 129}, 130: {'name': 'Swimwear', 'id': 130}, 131: {'name': 'Cabinetry', 'id': 131}, 132: {'name': 'Tent', 'id': 132}, 133: {'name': 'Umbrella', 'id': 133}, 134: {'name': 'Balloon', 'id': 134}, 135: {'name': 'Billboard', 'id': 135}, 136: {'name': 'Bookcase', 'id': 136}, 137: {'name': 'Computer keyboard', 'id': 137}, 138: {'name': 'Doll', 'id': 138}, 139: {'name': 'Dairy', 'id': 139}, 140: {'name': 'Bed', 'id': 140}, 141: {'name': 'Fedora', 'id': 141}, 142: {'name': 'Seafood', 'id': 142}, 143: {'name': 'Fountain', 'id': 143}, 144: {'name': 'Traffic sign', 'id': 144}, 145: {'name': 'Hiking equipment', 'id': 145}, 146: {'name': 'Television', 'id': 146}, 147: {'name': 'Salad', 'id': 147}, 148: {'name': 'Bee', 'id': 148}, 149: {'name': 'Coffee table', 'id': 149}, 150: {'name': 'Cattle', 'id': 150}, 151: {'name': 'Marine mammal', 'id': 151}, 152: {'name': 'Goose', 'id': 152}, 153: {'name': 'Curtain', 'id': 153}, 154: {'name': 'Kitchen & dining room table', 'id': 154}, 155: {'name': 'Home appliance', 'id': 155}, 156: {'name': 'Marine invertebrates', 'id': 156}, 157: {'name': 'Countertop', 'id': 157}, 158: {'name': 'Office supplies', 'id': 158}, 159: {'name': 'Luggage and bags', 'id': 159}, 160: {'name': 'Lighthouse', 'id': 160}, 161: {'name': 'Cocktail', 'id': 161}, 162: {'name': 'Maple', 'id': 162}, 163: {'name': 'Saucer', 'id': 163}, 164: {'name': 'Paddle', 'id': 164}, 165: {'name': 'Bronze sculpture', 'id': 165}, 166: {'name': 'Beetle', 'id': 166}, 167: {'name': 'Box', 'id': 167}, 168: {'name': 'Necklace', 'id': 168}, 169: {'name': 'Monkey', 'id': 169}, 170: {'name': 'Whiteboard', 'id': 170}, 171: {'name': 'Plumbing fixture', 'id': 171}, 172: {'name': 'Kitchen appliance', 'id': 172}, 173: {'name': 'Plate', 'id': 173}, 174: {'name': 'Coffee', 'id': 174}, 175: {'name': 'Deer', 'id': 175}, 176: {'name': 'Surfboard', 'id': 176}, 177: {'name': 'Turtle', 'id': 177}, 178: {'name': 'Tool', 'id': 178}, 179: {'name': 'Handbag', 'id': 179}, 180: {'name': 'Football helmet', 'id': 180}, 181: {'name': 'Canoe', 'id': 181}, 182: {'name': 'Cart', 'id': 182}, 183: {'name': 'Scarf', 'id': 183}, 184: {'name': 'Beard', 'id': 184}, 185: {'name': 'Drawer', 'id': 185}, 186: {'name': 'Cowboy hat', 'id': 186}, 187: {'name': 'Clock', 'id': 187}, 188: {'name': 'Convenience store', 'id': 188}, 189: {'name': 'Sandwich', 'id': 189}, 190: {'name': 'Traffic light', 'id': 190}, 191: {'name': 'Spider', 'id': 191}, 192: {'name': 'Bread', 'id': 192}, 193: {'name': 'Squirrel', 'id': 193}, 194: {'name': 'Vase', 'id': 194}, 195: {'name': 'Rifle', 'id': 195}, 196: {'name': 'Cello', 'id': 196}, 197: {'name': 'Pumpkin', 'id': 197}, 198: {'name': 'Elephant', 'id': 198}, 199: {'name': 'Lizard', 'id': 199}, 200: {'name': 'Mushroom', 'id': 200}, 201: {'name': 'Baseball glove', 'id': 201}, 202: {'name': 'Juice', 'id': 202}, 203: {'name': 'Skirt', 'id': 203}, 204: {'name': 'Skull', 'id': 204}, 205: {'name': 'Lamp', 'id': 205}, 206: {'name': 'Musical keyboard', 'id': 206}, 207: {'name': 'High heels', 'id': 207}, 208: {'name': 'Falcon', 'id': 208}, 209: {'name': 'Ice cream', 'id': 209}, 210: {'name': 'Mug', 'id': 210}, 211: {'name': 'Watch', 'id': 211}, 212: {'name': 'Boot', 'id': 212}, 213: {'name': 'Ski', 'id': 213}, 214: {'name': 'Taxi', 'id': 214}, 215: {'name': 'Sunflower', 'id': 215}, 216: {'name': 'Pastry', 'id': 216}, 217: {'name': 'Tap', 'id': 217}, 218: {'name': 'Bowl', 'id': 218}, 219: {'name': 'Glove', 'id': 219}, 220: {'name': 'Parrot', 'id': 220}, 221: {'name': 'Eagle', 'id': 221}, 222: {'name': 'Tin can', 'id': 222}, 223: {'name': 'Platter', 'id': 223}, 224: {'name': 'Sandal', 'id': 224}, 225: {'name': 'Violin', 'id': 225}, 226: {'name': 'Penguin', 'id': 226}, 227: {'name': 'Sofa bed', 'id': 227}, 228: {'name': 'Frog', 'id': 228}, 229: {'name': 'Chicken', 'id': 229}, 230: {'name': 'Lifejacket', 'id': 230}, 231: {'name': 'Sink', 'id': 231}, 232: {'name': 'Strawberry', 'id': 232}, 233: {'name': 'Bear', 'id': 233}, 234: {'name': 'Muffin', 'id': 234}, 235: {'name': 'Swan', 'id': 235}, 236: {'name': 'Candle', 'id': 236}, 237: {'name': 'Pillow', 'id': 237}, 238: {'name': 'Owl', 'id': 238}, 239: {'name': 'Kitchen utensil', 'id': 239}, 240: {'name': 'Dragonfly', 'id': 240}, 241: {'name': 'Tortoise', 'id': 241}, 242: {'name': 'Mirror', 'id': 242}, 243: {'name': 'Lily', 'id': 243}, 244: {'name': 'Pizza', 'id': 244}, 245: {'name': 'Coin', 'id': 245}, 246: {'name': 'Cosmetics', 'id': 246}, 247: {'name': 'Piano', 'id': 247}, 248: {'name': 'Tomato', 'id': 248}, 249: {'name': 'Chest of drawers', 'id': 249}, 250: {'name': 'Teddy bear', 'id': 250}, 251: {'name': 'Tank', 'id': 251}, 252: {'name': 'Squash', 'id': 252}, 253: {'name': 'Lion', 'id': 253}, 254: {'name': 'Brassiere', 'id': 254}, 255: {'name': 'Sheep', 'id': 255}, 256: {'name': 'Spoon', 'id': 256}, 257: {'name': 'Dinosaur', 'id': 257}, 258: {'name': 'Tripod', 'id': 258}, 259: {'name': 'Tablet computer', 'id': 259}, 260: {'name': 'Rabbit', 'id': 260}, 261: {'name': 'Skateboard', 'id': 261}, 262: {'name': 'Snake', 'id': 262}, 263: {'name': 'Shellfish', 'id': 263}, 264: {'name': 'Sparrow', 'id': 264}, 265: {'name': 'Apple', 'id': 265}, 266: {'name': 'Goat', 'id': 266}, 267: {'name': 'French fries', 'id': 267}, 268: {'name': 'Lipstick', 'id': 268}, 269: {'name': 'studio couch', 'id': 269}, 270: {'name': 'Hamburger', 'id': 270}, 271: {'name': 'Tea', 'id': 271}, 272: {'name': 'Telephone', 'id': 272}, 273: {'name': 'Baseball bat', 'id': 273}, 274: {'name': 'Bull', 'id': 274}, 275: {'name': 'Headphones', 'id': 275}, 276: {'name': 'Lavender', 'id': 276}, 277: {'name': 'Parachute', 'id': 277}, 278: {'name': 'Cookie', 'id': 278}, 279: {'name': 'Tiger', 'id': 279}, 280: {'name': 'Pen', 'id': 280}, 281: {'name': 'Racket', 'id': 281}, 282: {'name': 'Fork', 'id': 282}, 283: {'name': 'Bust', 'id': 283}, 284: {'name': 'Miniskirt', 'id': 284}, 285: {'name': 'Sea lion', 'id': 285}, 286: {'name': 'Egg', 'id': 286}, 287: {'name': 'Saxophone', 'id': 287}, 288: {'name': 'Giraffe', 'id': 288}, 289: {'name': 'Waste container', 'id': 289}, 290: {'name': 'Snowboard', 'id': 290}, 291: {'name': 'Wheelchair', 'id': 291}, 292: {'name': 'Medical equipment', 'id': 292}, 293: {'name': 'Antelope', 'id': 293}, 294: {'name': 'Harbor seal', 'id': 294}, 295: {'name': 'Toilet', 'id': 295}, 296: {'name': 'Shrimp', 'id': 296}, 297: {'name': 'Orange', 'id': 297}, 298: {'name': 'Cupboard', 'id': 298}, 299: {'name': 'Wall clock', 'id': 299}, 300: {'name': 'Pig', 'id': 300}, 301: {'name': 'Nightstand', 'id': 301}, 302: {'name': 'Bathroom accessory', 'id': 302}, 303: {'name': 'Grape', 'id': 303}, 304: {'name': 'Dolphin', 'id': 304}, 305: {'name': 'Lantern', 'id': 305}, 306: {'name': 'Trumpet', 'id': 306}, 307: {'name': 'Tennis racket', 'id': 307}, 308: {'name': 'Crab', 'id': 308}, 309: {'name': 'Sea turtle', 'id': 309}, 310: {'name': 'Cannon', 'id': 310}, 311: {'name': 'Accordion', 'id': 311}, 312: {'name': 'Door handle', 'id': 312}, 313: {'name': 'Lemon', 'id': 313}, 314: {'name': 'Foot', 'id': 314}, 315: {'name': 'Mouse', 'id': 315}, 316: {'name': 'Wok', 'id': 316}, 317: {'name': 'Volleyball', 'id': 317}, 318: {'name': 'Pasta', 'id': 318}, 319: {'name': 'Earrings', 'id': 319}, 320: {'name': 'Banana', 'id': 320}, 321: {'name': 'Ladder', 'id': 321}, 322: {'name': 'Backpack', 'id': 322}, 323: {'name': 'Crocodile', 'id': 323}, 324: {'name': 'Roller skates', 'id': 324}, 325: {'name': 'Scoreboard', 'id': 325}, 326: {'name': 'Jellyfish', 'id': 326}, 327: {'name': 'Sock', 'id': 327}, 328: {'name': 'Camel', 'id': 328}, 329: {'name': 'Plastic bag', 'id': 329}, 330: {'name': 'Caterpillar', 'id': 330}, 331: {'name': 'Sushi', 'id': 331}, 332: {'name': 'Whale', 'id': 332}, 333: {'name': 'Leopard', 'id': 333}, 334: {'name': 'Barrel', 'id': 334}, 335: {'name': 'Fireplace', 'id': 335}, 336: {'name': 'Stool', 'id': 336}, 337: {'name': 'Snail', 'id': 337}, 338: {'name': 'Candy', 'id': 338}, 339: {'name': 'Rocket', 'id': 339}, 340: {'name': 'Cheese', 'id': 340}, 341: {'name': 'Billiard table', 'id': 341}, 342: {'name': 'Mixing bowl', 'id': 342}, 343: {'name': 'Bowling equipment', 'id': 343}, 344: {'name': 'Knife', 'id': 344}, 345: {'name': 'Loveseat', 'id': 345}, 346: {'name': 'Hamster', 'id': 346}, 347: {'name': 'Mouse', 'id': 347}, 348: {'name': 'Shark', 'id': 348}, 349: {'name': 'Teapot', 'id': 349}, 350: {'name': 'Trombone', 'id': 350}, 351: {'name': 'Panda', 'id': 351}, 352: {'name': 'Zebra', 'id': 352}, 353: {'name': 'Mechanical fan', 'id': 353}, 354: {'name': 'Carrot', 'id': 354}, 355: {'name': 'Cheetah', 'id': 355}, 356: {'name': 'Gondola', 'id': 356}, 357: {'name': 'Bidet', 'id': 357}, 358: {'name': 'Jaguar', 'id': 358}, 359: {'name': 'Ladybug', 'id': 359}, 360: {'name': 'Crown', 'id': 360}, 361: {'name': 'Snowman', 'id': 361}, 362: {'name': 'Bathtub', 'id': 362}, 363: {'name': 'Table tennis racket', 'id': 363}, 364: {'name': 'Sombrero', 'id': 364}, 365: {'name': 'Brown bear', 'id': 365}, 366: {'name': 'Lobster', 'id': 366}, 367: {'name': 'Refrigerator', 'id': 367}, 368: {'name': 'Oyster', 'id': 368}, 369: {'name': 'Handgun', 'id': 369}, 370: {'name': 'Oven', 'id': 370}, 371: {'name': 'Kite', 'id': 371}, 372: {'name': 'Rhinoceros', 'id': 372}, 373: {'name': 'Fox', 'id': 373}, 374: {'name': 'Light bulb', 'id': 374}, 375: {'name': 'Polar bear', 'id': 375}, 376: {'name': 'Suitcase', 'id': 376}, 377: {'name': 'Broccoli', 'id': 377}, 378: {'name': 'Otter', 'id': 378}, 379: {'name': 'Mule', 'id': 379}, 380: {'name': 'Woodpecker', 'id': 380}, 381: {'name': 'Starfish', 'id': 381}, 382: {'name': 'Kettle', 'id': 382}, 383: {'name': 'Jet ski', 'id': 383}, 384: {'name': 'Window blind', 'id': 384}, 385: {'name': 'Raven', 'id': 385}, 386: {'name': 'Grapefruit', 'id': 386}, 387: {'name': 'Chopsticks', 'id': 387}, 388: {'name': 'Tart', 'id': 388}, 389: {'name': 'Watermelon', 'id': 389}, 390: {'name': 'Cucumber', 'id': 390}, 391: {'name': 'Infant bed', 'id': 391}, 392: {'name': 'Missile', 'id': 392}, 393: {'name': 'Gas stove', 'id': 393}, 394: {'name': 'Bathroom cabinet', 'id': 394}, 395: {'name': 'Beehive', 'id': 395}, 396: {'name': 'Alpaca', 'id': 396}, 397: {'name': 'Doughnut', 'id': 397}, 398: {'name': 'Hippopotamus', 'id': 398}, 399: {'name': 'Ipod', 'id': 399}, 400: {'name': 'Kangaroo', 'id': 400}, 401: {'name': 'Ant', 'id': 401}, 402: {'name': 'Bell pepper', 'id': 402}, 403: {'name': 'Goldfish', 'id': 403}, 404: {'name': 'Ceiling fan', 'id': 404}, 405: {'name': 'Shotgun', 'id': 405}, 406: {'name': 'Barge', 'id': 406}, 407: {'name': 'Potato', 'id': 407}, 408: {'name': 'Jug', 'id': 408}, 409: {'name': 'Microwave oven', 'id': 409}, 410: {'name': 'Bat', 'id': 410}, 411: {'name': 'Ostrich', 'id': 411}, 412: {'name': 'Turkey', 'id': 412}, 413: {'name': 'Sword', 'id': 413}, 414: {'name': 'Tennis ball', 'id': 414}, 415: {'name': 'Pineapple', 'id': 415}, 416: {'name': 'Closet', 'id': 416}, 417: {'name': 'Stop sign', 'id': 417}, 418: {'name': 'Taco', 'id': 418}, 419: {'name': 'Pancake', 'id': 419}, 420: {'name': 'Hot dog', 'id': 420}, 421: {'name': 'Organ', 'id': 421}, 422: {'name': 'Rays and skates', 'id': 422}, 423: {'name': 'Washing machine', 'id': 423}, 424: {'name': 'Waffle', 'id': 424}, 425: {'name': 'Snowplow', 'id': 425}, 426: {'name': 'Koala', 'id': 426}, 427: {'name': 'Honeycomb', 'id': 427}, 428: {'name': 'Sewing machine', 'id': 428}, 429: {'name': 'Horn', 'id': 429}, 430: {'name': 'Frying pan', 'id': 430}, 431: {'name': 'Seat belt', 'id': 431}, 432: {'name': 'Zucchini', 'id': 432}, 433: {'name': 'Golf cart', 'id': 433}, 434: {'name': 'Pitcher', 'id': 434}, 435: {'name': 'Fire hydrant', 'id': 435}, 436: {'name': 'Ambulance', 'id': 436}, 437: {'name': 'Golf ball', 'id': 437}, 438: {'name': 'Tiara', 'id': 438}, 439: {'name': 'Raccoon', 'id': 439}, 440: {'name': 'Belt', 'id': 440}, 441: {'name': 'Corded phone', 'id': 441}, 442: {'name': 'Swim cap', 'id': 442}, 443: {'name': 'Red panda', 'id': 443}, 444: {'name': 'Asparagus', 'id': 444}, 445: {'name': 'Scissors', 'id': 445}, 446: {'name': 'Limousine', 'id': 446}, 447: {'name': 'Filing cabinet', 'id': 447}, 448: {'name': 'Bagel', 'id': 448}, 449: {'name': 'Wood-burning stove', 'id': 449}, 450: {'name': 'Segway', 'id': 450}, 451: {'name': 'Ruler', 'id': 451}, 452: {'name': 'Bow and arrow', 'id': 452}, 453: {'name': 'Balance beam', 'id': 453}, 454: {'name': 'Kitchen knife', 'id': 454}, 455: {'name': 'Cake stand', 'id': 455}, 456: {'name': 'Banjo', 'id': 456}, 457: {'name': 'Flute', 'id': 457}, 458: {'name': 'Rugby ball', 'id': 458}, 459: {'name': 'Dagger', 'id': 459}, 460: {'name': 'Dog bed', 'id': 460}, 461: {'name': 'Cabbage', 'id': 461}, 462: {'name': 'Picnic basket', 'id': 462}, 463: {'name': 'Peach', 'id': 463}, 464: {'name': 'Submarine sandwich', 'id': 464}, 465: {'name': 'Pear', 'id': 465}, 466: {'name': 'Lynx', 'id': 466}, 467: {'name': 'Pomegranate', 'id': 467}, 468: {'name': 'Shower', 'id': 468}, 469: {'name': 'Blue jay', 'id': 469}, 470: {'name': 'Printer', 'id': 470}, 471: {'name': 'Hedgehog', 'id': 471}, 472: {'name': 'Coffeemaker', 'id': 472}, 473: {'name': 'Worm', 'id': 473}, 474: {'name': 'Drinking straw', 'id': 474}, 475: {'name': 'Remote control', 'id': 475}, 476: {'name': 'Radish', 'id': 476}, 477: {'name': 'Canary', 'id': 477}, 478: {'name': 'Seahorse', 'id': 478}, 479: {'name': 'Wardrobe', 'id': 479}, 480: {'name': 'Toilet paper', 'id': 480}, 481: {'name': 'Centipede', 'id': 481}, 482: {'name': 'Croissant', 'id': 482}, 483: {'name': 'Snowmobile', 'id': 483}, 484: {'name': 'Burrito', 'id': 484}, 485: {'name': 'Porcupine', 'id': 485}, 486: {'name': 'Cutting board', 'id': 486}, 487: {'name': 'Dice', 'id': 487}, 488: {'name': 'Harpsichord', 'id': 488}, 489: {'name': 'Perfume', 'id': 489}, 490: {'name': 'Drill', 'id': 490}, 491: {'name': 'Calculator', 'id': 491}, 492: {'name': 'Willow', 'id': 492}, 493: {'name': 'Pretzel', 'id': 493}, 494: {'name': 'Guacamole', 'id': 494}, 495: {'name': 'Popcorn', 'id': 495}, 496: {'name': 'Harp', 'id': 496}, 497: {'name': 'Towel', 'id': 497}, 498: {'name': 'Mixer', 'id': 498}, 499: {'name': 'Digital clock', 'id': 499}, 500: {'name': 'Alarm clock', 'id': 500}, 501: {'name': 'Artichoke', 'id': 501}, 502: {'name': 'Milk', 'id': 502}, 503: {'name': 'Common fig', 'id': 503}, 504: {'name': 'Power plugs and sockets', 'id': 504}, 505: {'name': 'Paper towel', 'id': 505}, 506: {'name': 'Blender', 'id': 506}, 507: {'name': 'Scorpion', 'id': 507}, 508: {'name': 'Stretcher', 'id': 508}, 509: {'name': 'Mango', 'id': 509}, 510: {'name': 'Magpie', 'id': 510}, 511: {'name': 'Isopod', 'id': 511}, 512: {'name': 'Personal care', 'id': 512}, 513: {'name': 'Unicycle', 'id': 513}, 514: {'name': 'Punching bag', 'id': 514}, 515: {'name': 'Envelope', 'id': 515}, 516: {'name': 'Scale', 'id': 516}, 517: {'name': 'Wine rack', 'id': 517}, 518: {'name': 'Submarine', 'id': 518}, 519: {'name': 'Cream', 'id': 519}, 520: {'name': 'Chainsaw', 'id': 520}, 521: {'name': 'Cantaloupe', 'id': 521}, 522: {'name': 'Serving tray', 'id': 522}, 523: {'name': 'Food processor', 'id': 523}, 524: {'name': 'Dumbbell', 'id': 524}, 525: {'name': 'Jacuzzi', 'id': 525}, 526: {'name': 'Slow cooker', 'id': 526}, 527: {'name': 'Syringe', 'id': 527}, 528: {'name': 'Dishwasher', 'id': 528}, 529: {'name': 'Tree house', 'id': 529}, 530: {'name': 'Briefcase', 'id': 530}, 531: {'name': 'Stationary bicycle', 'id': 531}, 532: {'name': 'Oboe', 'id': 532}, 533: {'name': 'Treadmill', 'id': 533}, 534: {'name': 'Binoculars', 'id': 534}, 535: {'name': 'Bench', 'id': 535}, 536: {'name': 'Cricket ball', 'id': 536}, 537: {'name': 'Salt and pepper shakers', 'id': 537}, 538: {'name': 'Squid', 'id': 538}, 539: {'name': 'Light switch', 'id': 539}, 540: {'name': 'Toothbrush', 'id': 540}, 541: {'name': 'Spice rack', 'id': 541}, 542: {'name': 'Stethoscope', 'id': 542}, 543: {'name': 'Winter melon', 'id': 543}, 544: {'name': 'Ladle', 'id': 544}, 545: {'name': 'Flashlight', 'id': 545}}


def replace_name(nm):
    if nm == 'Face':
        return 'Human face'
    if nm == 'Hair':
        return 'Human hair'
    if nm == 'Head':
        return 'Human head'
    if nm == 'Bike':
        return 'Bicycle'
    if nm == 'Arm':
        return 'Human arm'
    if nm == 'Nose':
        return 'Human nose'
    if nm == 'Eye':
        return 'Human eye'
    if nm == 'Mouth':
        return 'Human mouth'
    if nm == 'Hand':
        return 'Human hand'
    if nm == 'Leg':
        return 'Human leg'
    if nm == 'Licence plate':
        return 'Vehicle registration plate'
    if nm == 'Ear':
        return 'Human ear'
    if nm == 'Office':
        return 'Office building'
    if nm == 'Beard':
        return 'Human beard'
    if nm == 'studio couch':
        return 'Studio couch'
    if nm == 'Foot':
        return 'Human foot'
    return nm


def get_class_name_mappings():
    d1, d2 = get_description_for_labels()
    tensorflow_classes_1 = dict()
    tensorflow_classes_2 = dict()
    for id in TENSORFLOW_CATEGORIES:
        name = TENSORFLOW_CATEGORIES[id]['name']
        name = replace_name(name)
        if name in d2:
            tensorflow_classes_1[id] = d2[name]
            tensorflow_classes_2[d2[name]] = id
        else:
            print(name, 'fail')
            exit()
    return tensorflow_classes_1, tensorflow_classes_2


def get_real_annotations(table, classes):
    res = dict()
    for i, label in enumerate(classes):
        # print('Go for {} {}'.format(i, label))
        part = table[table['LabelName'] == label].copy()
        ids = part['ImageID'].values
        xmin = part['XMin'].values
        xmax = part['XMax'].values
        ymin = part['YMin'].values
        ymax = part['YMax'].values

        for i in range(len(ids)):
            id = ids[i]
            if id not in res:
                res[id] = dict()
            if label not in res[id]:
                res[id][label] = []
            box = [xmin[i], xmax[i], ymin[i], ymax[i]]
            res[id][label].append(box)
    return res


def get_detections(boxes_files):
    res = dict()
    tc1, tc2 = get_class_name_mappings()
    for f in boxes_files:
        # print('Go for {}'.format(f))
        id = os.path.basename(f)[:-5]
        scale, output_dict = load_from_file(f)
        if id not in res:
            res[id] = dict()
        num_detections = output_dict['num_detections']
        classes = output_dict['detection_classes']
        boxes = output_dict['detection_boxes']
        scores = output_dict['detection_scores']

        for i in range(num_detections):
            label = tc1[classes[i]]
            if label not in res[id]:
                res[id][label] = []
            box = [boxes[i][1], boxes[i][3], boxes[i][0], boxes[i][2], scores[i]]
            res[id][label].append(box)

    return res


def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.

    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def find_validation_score():
    from keras_retinanet.utils.anchors import compute_overlap

    iou_threshold = 0.5
    tc1, tc2 = get_class_name_mappings()
    d1, d2 = get_description_for_labels()
    all_files = glob.glob(DATASET_PATH + 'validation_big/*.jpg')
    all_ids = []
    for a in all_files:
        all_ids.append(os.path.basename(a)[:-4])
    print('Total image files: {} {}'.format(len(all_files), len(all_ids)))
    valid = pd.read_csv(DATASET_PATH + 'annotations/validation-annotations-bbox.csv')
    print('Number of files in annotations: {}'.format(len(valid['ImageID'].unique())))
    boxes_files = glob.glob(OUTPUT_PATH + 'cache_tensorflow_validation/*.pklz')
    print('Predictions found: {}'.format(len(boxes_files)))
    unique_classes = valid['LabelName'].unique()
    print('Unique classes: {}'.format(len(unique_classes)))

    print('Read detections...')
    all_detections = get_detections(boxes_files)
    print('Read annotations...')
    all_annotations = get_real_annotations(valid, unique_classes)

    average_precisions = {}
    for zz, label in enumerate(unique_classes):
        print('Go for: {} ({})'.format(d1[label], label))

        if label not in tc2:
            average_precisions[label] = 0, 1
            continue

        false_positives = np.zeros((0,))
        true_positives  = np.zeros((0,))
        scores          = np.zeros((0,))
        num_annotations = 0.0

        for i in range(len(all_ids)):
            detections = []
            annotations = []
            id = all_ids[i]
            if id in all_detections:
                if label in all_detections[id]:
                    detections = all_detections[id][label]
            if id in all_annotations:
                if label in all_annotations[id]:
                    annotations = all_annotations[id][label]

            if len(detections) == 0 and len(annotations) == 0:
                continue

            if 0:
                if len(detections) > 0 and len(annotations) > 0:
                    print(detections)
                    print(annotations)
                    print('----')

            num_annotations += len(annotations)
            detected_annotations = []

            for d in detections:
                scores = np.append(scores, d[4])

                if len(annotations) == 0:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)
                    continue

                overlaps            = compute_overlap(np.expand_dims(np.array(d, dtype=np.float64), axis=0), np.array(annotations, dtype=np.float64))
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap         = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives  = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)

        if num_annotations == 0:
            average_precisions[label] = 0, 0
            continue

        # sort by score
        indices = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives = true_positives[indices]

        # compute false positives and true positives
        false_positives = np.cumsum(false_positives)
        true_positives = np.cumsum(true_positives)

        # compute recall and precision
        recall = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

        # compute average precision
        average_precision = _compute_ap(recall, precision)
        average_precisions[label] = average_precision, num_annotations
        print(d1[label], average_precision, num_annotations)

    present_classes = 0
    precision = 0
    for label, (average_precision, num_annotations) in average_precisions.items():
        print('{:.0f} instances of class'.format(num_annotations), d1[label], 'with average precision: {:.4f}'.format(average_precision))
        if num_annotations > 0:
            present_classes += 1
            precision += average_precision
    mean_ap = precision / present_classes
    print('mAP: {}'.format(mean_ap))


if __name__ == '__main__':
    find_validation_score()


'''
mAP validation: 0.3535536 
mAP with absent: 0.340904
'''