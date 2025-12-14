from ..data_aug import pil_image_to_tensor
from ..ab_dataset import ABDataset
from ..dataset_split import train_val_test_split
from ..dataset_cache import get_dataset_cache_path, read_cached_dataset_status, cache_dataset_status
# from .mm_image_folder import MMImageFolder
from ..dataset_split import train_val_split
# from torchvision.datasets import CIFAR10 as RawCIFAR10
import os
from typing import Dict, List, Optional
from torchvision.transforms import Compose, Resize
from utils.common.others import HiddenPrints
import numpy as np
from ..registery import dataset_register
import torch
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image
from utils.common.log import logger
from utils.common.data_record import read_json
import cv2

all_classes = ['net', 'pitcher', 'orange', 'yes', 'white', 'skiing', 'red', 'frisbee', 'brushing teeth', 'no', 'black and white', 'skateboard', '1', 'blue', 'green', 'motorcycle', 'gray', '2', 'purse', 'skis', 'poles', 'surfboard', 'dog', 'on', 'office', 'large', 'very big', 'laptop', 'vent', 'computer', 'black', 'bear', '3', 'wii', 'glasses', 'tree', 'eating', 'log', '5', 'left', 'living room', 'pink', 'right', 'railing', 'grass', 'wire', '10 years', 'knife', 'cake', 'banana', 'chef', 'vanilla', '4', 'outdoor', 'mustard', 'bun', 'clouds', 'dock', 'brown', 'silver', 'refrigerator', 'square', 'teddy', 'elm', 'stripes', 'baseball', 'catcher', 'beer', 'bottom', 'north', 'nike', 'yellow and white', 'morning', 'elephant', 'red and white', 'propeller', 'tan', 'wall', 'clock', 'table', '0', 'wood', 'christmas', 'spinach', 'thick', 'bag', 'leaves', 'necklace', '6', 'bathroom', 'shower', 'towel', 'solid', 'referee', 'wilson', 'e', '24', 'hat', 'grazing', 'sheep', '10', 'tag', 'spanish', 'hot dog', 'plate', 'lunch', 'butter', 'peppers', 'onions', 'very', 'pig', 'sweet', 'flowers', 'floral', 'yellow', 'window', '7', 'pizza', 'car', '', 'cargo', 'stairs', 'abstract', 'rug', 'baseball cap', 'texting', 'pole', 'crosswalk', 'nothing', 'urban', 'bus', 'light', 'afternoon', 'boat', 'cheese', 'paper', 'real', 'sun', 'birthday', 'words', 'inside', 'shadows', 'tomato', 'evergreen', '100 feet', 'trees', 'building', 'hay', 'ski pole', 'walking', 'ice', 'laundry', 'pepsi', 'good', '1:50', 'purple', '13', 'africa', 'teddy bears', 'socks', 'giraffe', 'soccer', 'blue and yellow', 'zebras', 'cupcake', 'broccoli', 'parking lot', 'cows', 'herding', 'on table', 'fish', 'nightstand', '50', 'overcast', 'cross', 'toaster oven', 'tile', '11:55', 'red and yellow', 'nowhere', 'hair dryer', 'truck', '11', 'people', 'rectangle', 'hot dogs', 'party', '12:55', 'apron', 'kitchen', 'cooking', 'ring', '1 way', 'stop', 'neither', 'many', 'female', 'brushing', 'tie', 'tennis racket', 'knife and fork', 'restaurant', 'cat', 'bed', 'sand', 'ocean', 'cold', 'kites', 'cumulus', 'standing', 'male', 'star', 'tracks', 'chocolate', 'round', 'fork and knife', 'yankees', 'pictures', 'dots', 'bird', 'parrot', 'red white and blue', 'man', 'metal', 'fence', 'snowboarding', 'pine', 'snow', 'shorts', 'swim', 'wine', 'brick', 'no parking', 'children', 'beef', 'phone', 'english', 'cell phone', 'pink and yellow', 'clear', 'watermelon', 'bedroom', 'fork', 'cow', 'rackets', 'tennis rackets', '8', 'collar', 'tennis', 'playing tennis', 'skirt', '30', 'polka dot', 'beach', 'horse', 'grill', 'african american', 'down', 'street', 'in air', 'sweater', 'yellow and blue', 'park', 'spectators', 'parasailing', '31', 'river', '55', 'shadow', 'winter', 'chicken', 'tea', 'evening', 'dusk', 'ski resort', 'helmet', 'bench', 'resting', 'elephants', 'southwest', 'usa', 'cars', 'town', 'bananas', 'umbrella', 'container', 'woman', 'on counter', 'salad', 'striped', 'motel', 'vertical', 'oranges', 'hot sauce', 'bottle', 'juice', 'eyes', 'ground', 'backpack', 'black and yellow', 'forward', 'jackets', '1 on right', 'green and yellow', 'playing baseball', 'riding', 'sitting', 'carrot', 'basket', 'seagull', 'ski poles', 'p', 'parking', 'street light', 'strap', 'bike', 'riding bike', 'poodle', 'shoes', 'carpet', 'lettuce', 'food', '1 foot', 'roses', 'mountains', 'scissors', 'camera', 'beige', 'beard', 'cutting', 'baby', 'tape', 'watch', 'never', 'taking picture', 'eggs', 'syrup', 'sandwich', 'water skiing', 'microphone', 'back', 'bears', 'donuts', 'w', 'sky', 'double decker', 'england', 'surfing', 'running', 'shirt', 'barn', 'weather vane', 'white and blue', 'fishing', 'bridge', 'los angeles', 'open', 'red sox', 'bat', 'plane', 'white and green', 'transportation', 'sunny', 'bus stop', 'city', 'brown and white', 'bicycle', 'crow', 'magazines', 'daisy', '14', 'old', 'curtains', 'snowboard', 'dinosaur', 'racing', 'asphalt', 'court', 'plastic', 'circle', 'red and blue', 'zebra', '12', 'biplane', 'shallow', 'brazil', 'logo', '2:20', 'electric', 'motion', 'toothbrushes', 'orange and white', '66', 'spoon', 'toyota', 'tennis shoes', '46', 'second', 'no 1', 'iphone', 'friend', 'apple', '15', 'tiger', 'glove', 'airplane', 'bow', 'air france', 'passengers', 'tv', 'on building', '3:55', 'victorian', 'steeple', 'happy', 'skateboarding', 'fruit', 'cutting board', 'cantaloupe', 'kiwi', 'sliced', 'heart', 'water', 'rainy', 'carrots', 'giraffes', 'eat', 'ramp', 'lab', 'field', 'horizontal', 'birds', 'home', 'shrimp', '12 feet', 'girl', 'modern', 'dell', 'boots', 'sunglasses', 'black and orange', 'yellow and black', 'gloves', 'hp', 'desk', 'both', 'sign', 'on street', '2000', 'cirrus', 'ceiling', 'fluorescent', 'up', '9', 'boys', 'playing soccer', 'american', 'passenger', 'turn', 'palm', 'wedding', 'branch', 'parrots', 'air force', 'on tracks', 'small', 'dirty', 'france', 'honda', '2.00', 'vase', 'flying', 'driving', 'tissue', 'protest', 'corona', 'twin', 'clothes', 't shirt', 'window sill', 'wild', 'noon', 'caution', 'spring', 'raining', 'cane', 'school', 'windsurfing', 'parachute', 'black and red', '25', 'background', 'toaster', 'planes', 'yellow and red', 'spatula', '10:10', 'ivory', 'train', 'highway', 'off', 'on track', 'electricity', 'italy', 'dinner', 'sink', 'squares', '5 ft', 'parked', 'store', 'dress', 'signs', 'football', 'rugby', 'stainless steel', 'dirt', 'blue and white', 'klm', 'house', 'unknown', 'ford', 'reading', 'chair', 'mountain', 'alive', 'water skis', 'picture', 'parade', 'trailer', 'boating', 'holding it', 'shade', 'cloth', 'candle', 'hose', 'hand', '3:25', 'on sidewalk', 'poster', 'downhill', 'reflection', 'summer', 'pickles', 'halloween', 'bats', 'london', 'zoo', 'surfer', 'racket', 'flickr', 'cutting hair', 'strawberries', 'mushroom', 'teddy bear', 'big', 'suitcase', 'veggie', 'pepper', 'houses', '70', 'toshiba', 'triangle', 'boxes', 'photograph', 'smoke', 'engine', 'camel', 'sidewalk', 'left 1', 'red and green', '4:35', 'on couch', 'candy', 'homemade', 'mouse', 'box', 'movie', '45', 'strawberry', 'fridge', 'full', 'vegetables', 'bright', 'play', 'remote', 'pond', 'savannah', 'celery', 'concrete', 'semi', 'scania', 'safety', 'posing', 'fabric', 'laying', 'couch', 'blueberries', 'handle', 'pipe', 'stick', 'steak', 'chain link', 'barbed wire', 'mozzarella', 'soda', 'fire hydrant', 'cat food', 'pepperoni', 'lot', 'licking', 'red and black', 'clay', 'tennis court', 'jumping', 'potatoes', 'toothbrush', 'kite', 'flying kite', 'broken', 'black and silver', 'lap', 'outside', '44', 'delta', 'greyhound', 'talking on phone', 'bad', 'kettle', '35', 'motorcycles', 'produce', 'steering wheel', '18', 'humans', 'coffee', 'white and brown', 'fall', 'bread', 'cherry', '4:30', 'flag', 'night', 'lamp', 'cucumber', 'porcelain', 'oval', 'museum', 'rain', 'sprinkles', '20', 'kids', 'bracelet', 'sneakers', 'mask', 'mickey mouse', 'very high', 'costume', 'cabbage', 'paint', 'lighting', 'young', 'air conditioner', 'wooden', 'board', 'beets', '16', 'lights', 'ladder', 'glass', 'fries', 'steamed', 'shepherd', 'cotton', 'suit', 'goatee', 'on his head', 'print', 'happy birthday', 'forks', 'travel', 'maple', '200', 'oil', 'jeans', 'can', 'chopsticks', 'on wall', 'construction', '36', 'chinese', 'festival', 'gas', 'throwing', 'circus', 'wires', 'not possible', 'plates', 'sugar', 'in', "women's", 'door', 'volleyball', 'serving', 'ponytail', 'business', 'decoration', 'santa', 'flat', 'barrel', '12:15', 'candles', 'free', 'hair', 'ball', 'stop sign', 'wetsuit', 'green and black', 'foreground', 'stands', 'china airlines', 'flower', '300', 'on bench', 'plaster', 'phones', 'sailboat', 'apples', 'road', 'recently', 'cones', 'cactus', 'rice', 'vegetarian', 'donut', 'ketchup', 'police', 'mirror', 'rock', 'meat', 'blinds', 'cell phones', 'china', 'rust', '7:25', 'stone', 'vans', 'middle', 'eagle', '9:30', 'ping pong', 'microwave', 'gmc', 'umbrellas', 'wrist', 'laughing', 'boy', 'next to toilet', 'tabby', 'petting', 'south', '40', 'checkered', 'slow', 'cardboard', 'windows', 'croissant', 'plain', 'cookie', 'on ground', 'low', 'water bottle', 'goggles', 'turkey', 'shut', 'kite flying', 'bowl', 'smile', 'in bowl', 'bush', 'cloudy', 'top left', 'skateboarder', 'coca cola', 'pan', 'drinking', 'short', 'floor', 'thanksgiving', 'radio', 'drink', 'on toilet', 'bike rack', 'bleachers', 'train tracks', 'horses', 'far', 'top', 'toilet', 'in water', 'private', 'nature', 'commercial', 'stroller', 'power', 'stuffed animals', 'uniforms', 'japan', 'faucet', 'green and orange', 'corn', 'white and yellow', 'mercedes', 'in sky', 'tarp', 'indian', 'counter', 'multicolored', 'polar', 'go', 'no number', 'swimming', 'bridle', 'cowboy', 'olives', 'pizza cutter', 'british airways', 'nighttime', 'australia', 'tiles', 'pug', 'wicker', 'british', 'us airways express', 'burton', 'christmas tree', 'napkin', 'writing', 'rocks', 'hello kitty', 'gold', 'fan', 'skateboards', 'day', 'on floor', '2008', 'dark', 'flying kites', 'rural', 'olympics', 'bmw', '34', 'denim', 'typing', 'for fun', 'steel', 'watching tv', 'driver', 'grapes', 'f', 'angels', 'roof', 'handlebars', 'train station', 'public', 'oak', 'sleeping', 'canada', 'air canada', 'on top', 'tired', 'blonde', 'cups', 'little', 'adidas', '10 feet', 'white and gray', 'leaf', 'fisheye', 'forest', 'war', 'octagon', 'raspberry', 'helmets', 'united states', '29', 'noodles', 'van', 'long', 'traveling', 'luggage', 'airport', 'single', 'pitching', 'dugout', 'garbage', 'happiness', 'cigarette', 'on tower', 'antelope', 'graffiti', 'skating', 'on road', 'curved', 'washington', 'ski lift', 'athletics', 'brace', 'squatting', 'catching', 'batter', 'batting', 'game', 'towards', '33', 'sliding', 'makeup', 'japanese', 'person', 'pirates', 'plaid', 'rose', 'daytime', 'keyboard', 'surfboards', 'hummingbird', 'ollie', '11:30', 'clock tower', 'san francisco', 'stopping', 'tags', 'samsung', 'computers', 'cabinets', 'talking', 'asparagus', '5 years', 'adult', 'rabbit', 'empty', 'softball', '1st', 'playing', 'chairs', 'farm', 'cross country', 'dump truck', 'women', 'snowboarder', 'tall', 'monkey', 'fire', 'books', 'cessna', 'chandelier', 'dunkin donuts', 'beans', 'relish', 'parking meter', 'ducks', 'sandals', 'doughnut', 'lighthouse', 'yacht', 'german shepherd', 'raw', 'chain', '2 feet', 'pedestal', 'mutt', 'race', 'poor', 'cat and dog', 'station', 'printer', 'daisies', 'front', 'gravel', 'grassy', 'pigeons', 'dogs', 'in car', 'life', 'wii remotes', 'suv', 'leather', 'bottom right', 'peace', 'blanket', 'frisbees', '12:30', 'scooter', 'going', 'analog', 'america', 'pitbull', 'relaxing', 'paddle boarding', 'white and pink', 'ride', 'side', 'on desk', 'on chair', '2012', 'multi', 'straight', 'big ben', 'closed', '3 feet', 'waves', 'buoy', 'trash can', 'medium', 'very tall', 'yamaha', 'sunlight', 'hit ball', 'dry', 'coke', 'gym', 'orange and black', 'center', 'rope', 'flip flops', 'siamese', 'crafts', 'color', 'italian', 'playing frisbee', 'skate park', 'orange juice', 'windowsill', 'thumb', 'pie', 'toast', 'no hat', 'benches', 'diamond', 'blender', 'avocado', 'television', 'speakers', 'pony', 'baseball field', 'pavement', 'not there', 'diamonds', '4 feet', 'goalie', 'soccer ball', 'runway', 'video game', 'gaming', 'casual', 'green and white', 'toilet brush', 'working', 'pickup', 'girls', 'remotes', 'pasta', 'hood', 'braves', 'skier', 'motorola', '17', 'b', '100', 'hospital', 'wagon', 'milk', 'ferry', 'rainbow', 'on bed', 'toward', '1:30', '19', 'mercedes benz', 'supreme', 'thin', 'platform', 'thai', 'storage', 'swan', 'peach', '10:05', 'dome', 'chiquita', '2:00', 'mountain dew', '23', 'knives', 'street sign', 'on beach', 'playing wii', 'stickers', 'yogurt', 'on grass', '9:45', 'gatorade', 'umpire', '37', 'desktop', 'desserts', 'main', 'boston', 'fell', 'top right', 'case', 'asleep', 'over', 'grapefruit', 'breakfast', 'headphones', 'freight', 'cup', 'sweatband', 'nobody', 'lamps', '9:25', 'scarf', 'on fridge', 'moving', 'fresh', 'blue jay', 'chihuahua', 'ceramic', 'mushrooms', 'on plate', 'human', 'power lines', 'hotel', 'map', 'earring', 'boarding', 'warm', 'napkins', 'brown and black', 'broom', 'basketball', 'papers', 'sad', 'kickstand', '60', 'shoulder', 'sleep', 'footprints', 'tunnel', '1990', 'hats', '6 inches', 'ham', 'bacon', 'church', '53', 'pineapple', 'at camera', 'red bull', 'pilot', 'tattoo', 'work', 'polar bear', 'taking off', 'website', '22', '4:00', 'coffee maker', 'fast', 'fur', 'rubber', 'tongs', 'german', 'germany', 'toy', '3:20', 'calm', 'pots', 'fruits', '9:20', 'drawer', 'oven', 'soup', 'stove', 'heels', 'wind', 'island', 'blood', 'leg', 'theater', 'tennis racquet', '21', 'gothic', '2:35', 'wii remote', 'turning', '20 feet', 'ears', 'fun', 'to right', 'child', 'fly', 'head', 'drywall', 'pier', 'feeding giraffe', 'in vase', 'burger', 'easter', 'onion', 'uniform', 'guitar', 'time', 'tomatoes', 'ship', 'tulips', 'glaze', 'tent', 'market', 'bandana', 'still', "don't know", 'piano', 'mouth', 'run', 'sparrow', 'lines', 'vest', '1950', 'jet', 'sepia', '2015', 'busy', 'dessert', '75', 'finch', 'pastries', 'outdoors', 'bakery', 'clean', 'ipod', 'tablecloth', 'looking at phone', 'in front', 'food truck', 'face', 'swinging', 'safari', '500', 'volkswagen', '2010', 'shelves', 'riding horses', '2016', 'towels', 'lemon', 'straw', 'bamboo', '5 feet', 'hardwood', 'h', 'meter', 'charging', 'bald', 'caucasian', 'man on left', 'stand', '27', 'dining room', 'sandwiches', '32', 'apartment', 'tower', 'virgin', 'out', 'white and red', "i don't know", 'chains', 'legs', 'goats', 's', 'dresser', 'camper', 'half', 'decorative', 'hawaiian', 'wheel', 'florida', 'reds', 'washington dc', 'moon', 'conference', 'screen', 'controller', 'robin', 'men', 'protection', 'harley davidson', 'coal', 'mustache', 'smiling', 'pedestrians', 'me', 'tray', 'monitor', 'bell', 'landscape', 'club', 'toothpick', 'seagulls', 'bowtie', 'lake', 'steam', 'surf', 'baseball glove', 'blinders', 'woods', 'shearing', 'dad', 'mixer', 'pot', 'blending', 'identification', 'owl', 'wine glass', 'new york', 'yarn', 'tennis ball', 'ice cream', 'chevrolet', 'shirt and tie', 'taking selfie', 'blue and green', "he isn't", 'cutting cake', 'east', 'setting', '7 eleven', 'stars', 'jockey', 'jacket', 'book', 'gray and white', 'pen', 'red white blue', 'above', 'alaska', 'tongue', 'feathers', 'k', 'camping', 'corner', 'away', 'ski', 'texas', 'fire truck', 'sailboats', 'jump', 'walk', 'spray paint', 'loading', 'united', '1000', 'roman numerals', 'surprise', '3rd', 'first', 'side of road', 'dodgers', 'airplanes', 'unsure', 'russian', 'wet', '5 star', 'blankets', 'natural', 'across street', 'smartphone', 'duck', 'sausage', 'paris', 'newspaper', 'pants', 'spices', 'pillow', 'to left', 'snowboards', 'colgate', 'on elephant', 'string', 'horns', '2:40', "men's", 'cobblestone', 'regular', 'staring', '28', 'barber shop', 'cut', 'x', 'above sink', 'above stove', 'dishes', 'dalmatian', 'watching', 'glazed', '5:25', 'messy', 'wallet', 'tuna', 'grilled', 'french', 'green and blue', 'sunflowers', 'wool', 'cabinet', 'shell', 'foil', 'bottles', 'bar', 'king', 'paper towels', 'friends', 'beagle', 'school bus', 'laptops', 'snowing', 'cement', 'pc', 'accident', 'stuffed animal', 'balance', 'white and black', 'cleats', 'on sink', 'pool', 'mom', 'downtown', 'asian', 'heater', 'bathing', '193', 'against wall', 'canopy', 'berries', 'military', 'pickle', 'clams', 'seafood', 'in box', 'boats', 'lizard', 'lemonade', 'm', 'soft', 'country', 'for sale', 'arm', 'listening', 'curly', 'play tennis', 'hands', 'cereal', 'blue and red', 'robe', 'soap', 'trains', 'throwing frisbee', 'smoking', 'india', 'headband', 'not very', 'westin', 'serve', 'bicycles', "can't tell", 'visibility', 'ana', 'reins', 'rodeo', 'riding motorcycle', 'mexico', 'mother', 'african', 'left and right', 'button', 'earrings', 'blackberry', 'cell', '10:00', 'harness', 'pillows', 'vegetable', 'tablet', 'fern', 'cats', 'golden retriever', 'goat', 'tractor', "valentine's day", 'hearts', 'khaki', 'man on right', "mcdonald's", 'arriving', 'husky', 'on skateboard', 'vases', 'coat', 'beanie', 'coming', 'granite', 'sports', 'leash', 'balls', 'blurry', 'baseball bat', 'mug', 'eiffel tower', 'worms', 'trash', 'terrier', 'painting', 'rooster', '42', 'jones', 'state farm', 'balloon', 'trunk', 'coach', 't', 'playing game', 'fireplace', 'behind clouds', 'uphill', 'motocross', 'sony', 'magazine', 'kitesurfing', 'catching frisbee', 'catch frisbee', 'bud light', 'fighting', '1 on left', 'very old', 'hallway', 'lexus', 'wii controller', '5:45', 'catholic', 'muffin', 'traffic light', 'grocery', 'shelf', '2:25', 'honey', 'plants', 'oars', 'foggy', "nathan's", 'cord', 'yard', '48', 'chimney', 'calico', 'suits', 'sideways', 'animals', 'black and blue', 'bikini', 'photographer', 'queen', '1:00', '12:05', 'horseback riding', 'awake', 'bunny', '12:00', 'continental', 'rye', 'family', 'lots', 'owner', 'palm tree', 'design', 'far right', 'tire', 'younger', 'biking', 'giants', 'caramel', 'polo', 'emirates', 'magnets', 'mat', 'ivy', 'cakes', 'bob', 'asia', 'graduation', 'cauliflower', 'c', 'rough', 'air', 'windy', 'victoria', 'trick', 'labrador', 'on left', 'yellow and green', 'butterfly', 'fake', 'on napkin', 'bricks', 'wine glasses', 'detroit', "man's", 'parsley', 'art', 'subway', 'wave', 'placemat', 'hydrant', 'sofa', 'pigeon', 'all', 'branches', 'plant', 'to eat', 'zucchini', 'feta', 'mouse pad', 'cloud', 'toilet paper', 'pumpkin', 'rowing', 'handicap', 'seeds', 'fly kite', 'chicago', 'marble', 'frame', '150', 'rocky', 'sauce', "it's not", 'control', 'high chair', 'playstation', 'xbox', 'roman', 'land', '1:35', 'lifeguard', 'size', 'bull', 'goose', '8 feet', 'recessed', 'statue', 'index', 'phillies', 'strike', 'mirrors', 'pointing', 'farmer', 'collie', 'motorbike', 'lanes', 'bikes', 'gas station', 'logs', 'smaller', 'desert', 'yield', 'flags', 'stool', 'kitten', 'doll', 'daffodils', 'letters', 'dishwasher', 'nuts', '2013', 'persian', 'swim trunks', 'deep', 'doubles', 'in field', 'wristband', 'wheels', 'baking', '4:15', '11:00', 'ear', '2007', '51', 'frog', 'boogie board', 'hungry', 'by window', 'ambulance', 'pigtails', 'microsoft', 'on man', 'laying down', '3:00', 'taxi', 'pedestrian', 'landing', 'numbers', '38', 'stones', 'clocks', 'new', 'picnic', 'fog', 'buffalo', 'under armour', 'orioles', 'bags', 'golden gate', 'castle', 'canoe', 'selfie', 'cream', 'floating', 'indoor', 'antique', 'aluminum', 'peas', 'sun hat', 'on right', 'flour', 'under sink', 'fashion', 'fedora', 'shells', '1 hour', 'puppy', 'motor', '120', 'sail', 'mexican', 'dead end', 'paddle', 'shop', 'boxing', 'birthday cake', 'chalk', 'style', 'nissan', 'sticker', 'north face', 'squash', 'not sure', 'seat', 'himself', 'circles', 'san diego', 'kia', 'mattress', 'obama', 'lamb', 'american flag', 'climbing', 'skull and crossbones', 'roast beef', 'visor', 'double', '52', 'high', 'stagecoach', 'cart', 'feeding', 'eaten', 'cone', 'smoothie', 'golf', 'colorado', 'electronics', '5:15', 'bowling', 'players', 'ketchup and mustard', 'styrofoam', '6 feet', 'hawk', 'cheddar', 'arabic', 'shower curtain', 'army', 'salmon', 'hanging', 'whole', 'behind fence', 'bars', 'moss', 'no dog', 'traffic', 'r', 'countryside', 'directions', 'cooked', 'aa', '6:45', '4 way', 'stripe', 'brand', 'baseball player', 'bunk', 'coleslaw', 'europe', 'dead', 'arch', 'scrambled', 'clothing', 'closet', 'egg', 'suitcases', 'indoors', 'tires', 'lilies', 'cafe', 'toothpaste', 'in background', 'tarmac', 'painted', 'sunset', 'orange and yellow', 'zebra and giraffe', 'ladybug', 'hills', 'tail', 'couple', 'kawasaki', 'smooth', 'powdered sugar', 'pedestrian crossing', 'french fries', 'teeth', 'ribbon', 'saddle', 'on train', '39', 'curb', 'tow', 'shark', 'white and orange', 'gravy', 'curtain', 'lime', 'skull', 'crossing', 'peacock', 'neck', 'hit', 'dragon', 'tissues', 'basil', 'waving', 'helicopter', 'mud', 'us', 'red and gray', 'sunflower', 'wallpaper', '11:20', 'seattle', 'bookshelf', 'looking', '1 inch', 'harley', 'urinal', 'navy', 'fedex', 'rays', 'deck', 'coaster', '1:20', '4:20', '5:00', 'jp morgan', 'palm trees', 'tub', 'pens', '2 people', 'speaker', 'hamburger', 'green beans', "it isn't", '10:20', 'buildings', 'on shelf', 'orange and blue', '90', 'north america', 'arrow', 'news', 'tropicana', 'formal', 'in grass', 'thumbs up', 'clip', 'tennis player', 'pastry', 'nose', 'pacifier', '11:35', 'different teams', 'cardinals', 'bagel', 'huge', 'out of focus', 'cook', 'wheat', 'photo', 'sedan', 'lanyard', 'pink and white', 'sesame', 'space', 'warning', 'snowy', 'tater tots', 'tropical', 'grandfather', 'mac', 'pajamas', '350', 'casserole', 'pelican', '2009', 'clydesdale', 'tow truck', 'belt', 'west', 'omelet', 'heavy', 'crown', 'in corner', 'hexagon', 'mound', 'iris', 'g', '2:15', '3:10', 'drawing', 'only', 'washing', 'nokia', 'windsor', 'icing', 'several', 'no smoking', 'kayak', 'frosting', 'jetblue', 'shoe', 'britain', 'ties', 'bank', 'camouflage', 'privacy', 'bib', 'blue and gray', 'looking out window', 'falling', 'bucket', 'cupcakes', 'throw ball', 'garden', 'almonds', 'starbucks', 'all way', 'home plate', 'base', 'toys', '1 in front', 'foot', 'california', 'towing', 'cheesecake', 'bushes', 'bow tie', 'down street', '2011', 'police officer', 'windmill', 'taking pictures', 'cleaning', 'on pole', 'main street', 'catch ball', 'mario', 'track', 'garage', "they aren't", 'tents', 'tattoos', '2:45', 'wheelchair', 'money', 'top hat', 'willow', 'brushing hair', '80', 'green and red', 'barrier', 'hiking', 'tank top', 'lufthansa', 'menu', 'forehand', 'wii controllers', 'hundreds', 'water ski', 'furniture', 'paisley', 'pizza hut', 'hill', 'prom', 'tiara', 'students', 'information', 'hazy', 'canon', 'bird feeder', 'crane', 'dr pepper', 'logitech', '2:10', 'all of them', 'utensils', 'telephone', 'converse', 'bone', 'jeep', 'nursing', 'krispy kreme', 'ranch', 'polka dots', 'railroad crossing', 'shirts', 'feeder', 'above toilet', 'unclear', 'below', '43', 'spoons', 'calendar', 'mint', 'spiderman', 'lg', 'concert', 'coats', 'lady', 'dodge', 'flat screen', '10:30', 'music', 'polar bears', 'riding horse', 'cookies', 'hot', 'behind', 'dole', '26', 'pans', 'love', 'winnie pooh', 'copyright', '2 hours', 'snowsuit', 'kissing', 'backhand', 'swans', 'nintendo', 'direction', 'waiting', 'mohawk', 'rail', 'hoodie', 'feet', '106', '10:55', 'coins', 'mitt', 'room', 'adults', 'cameras', 'marker', 'sled', 'conductor', 'farmers market', 'toiletries', 'blue and black', 'sprite', 'bank of america', 'heat', 'emergency', 'hard', '41', '6:00', 'in his hand', 'cluttered', 'grizzly', 'not', 'in hand', 'under table', 'd', 'hitting ball', 'photography', 'intersection', 'backwards', 'crocs', 'chips', 'harry potter', 'hawaii', 'half full', 'carriage', 'curious', 'geese', 'pork', 'l', 'sidecar', 'penguin', 'to see', 'pocket', 'steps', 'cubs', 'junk', 'deer', 'ottoman', 'salt', 'condiments', '1:55', 'post', 'bulldog', 'notebook', 'no cat', 'jets', 'knee pads', 'throw frisbee', 'drinks', 'leopard', 'grape', 'wine tasting', 'baskets', 'santa hat', 'chest', 'sewing', 'on car', 'sony ericsson', 'peeing', 'tour', 'fire extinguisher', 'lemons', 'wiimote', 'guitar hero', 'stopped', 'library', 'blue and pink', 'choppy', 'sailing', 'brush', 'jelly', 'dairy queen', 'shaking hands', 'ge', 'tigers', 'tokyo', 'buses', 'pink and blue', 'singles', 'iron', "don't walk", 'classroom', 'harbor', 'residential', 'joshua', 'uk', 'burgers', 'lace', 'overalls', 'ram', 'dancing', '47', 'shed', 'lid', "he's not", 'amtrak', 'ostrich', 'bathtub', '2:50', 'mall', 'slow down', 'hammer time', 'octopus', 'crib', 'broadway', 'pottery', 'wavy', 'holding phone', 'tusks', 'dining', 'packing', 'thomas', 'budweiser', 'beijing', '11:10', 'wide', 'slope', 'black and gray', 'chili', 'siblings', 'kayaking', 'captivity', 'rack', 'panda', 'pelicans', 'genetics', 'not in service', 'v', 'on laptop', 'gone', 'tying tie', 'scale', 'lily', 'cool', 'n', 'toilets', 'tree branch', 'copper', '870', 'shopping', 'batman', 'black and brown', 'legos', 'drinking water', 'burrito', 'spiral', 'ibm', 'tools', 'cherries', 'maple leaf', 'vines', 'sushi', 'baker', 'globe', 'wireless', 'compaq', 'do not enter', '1:05', 'advertisement', 'movement', 'model', 'hammock', 'swing', 'sheet', 'google', 'right 1', 'haircut', 'exit', 'tim hortons', 'lego', 'cucumbers', 'potato', 'egg salad', 'controllers', 'upside down', 'lion', 'camo', 'dirt bike', 'playing video games', 'crates', 'horizontally', 'plunger', 'radiator', 'in basket', 'cap', 'living', 'briefcase', 'ascending', 'flip phone', '101', 'gun', 'foam', 'serious', 'pancakes', 'heineken', 'driveway', 'cleaner', 'delivery', 'commuter', 'apple and banana', 'chase', 'trucks', 'trunks', '64', 'slacks', 'skiers', 'carrot cake', 'holding', 'surfers', 'horse racing', 'orchid', 'leaving', 'pitch', 'crest', 'miami', 'bus station', 'take off', 'diesel', 'pm', 'wetsuits', '7:35', 'tie dye', 'baked', 'life jacket', 'grilled cheese', 'meatballs', 'monster', 'smiley face', 'keys', 'straight ahead', 'badminton', 'end', '5:05', '10:50', 'each other', 'weeds', 'tinkerbell', 'rottweiler', 'apartments', 'sweatshirt', 'shore', 'switzerland', '65', 'jar', 'skate', 'raspberries', 'singing', 'on bus', 'carnations', 'descending', 'hsbc', 'space needle', 'skatepark', 'kenmore', 'db', "baby's breath", 'shelter', '1980', 'no left turn', '9:05', 'pipes', 'donkey', 'mitsubishi', 'tell time', 'outfield', 'flip', 'stadium', 'heinz', 'distance', 'macaroni', 'on plane', 'triumph', '4:50', 'on stove', 'shih tzu', 'fried', 'sunrise', '2nd', 'suzuki', 'traffic lights', 'hitting', 'healthy', 'tulip', 'right side', 'on sign', 'maroon', '5:40', 'michigan', 'close', 'license plate', 'sniffing', '1:15', 'cardinal', 'older', 'nest', 'colored', 'in back', 'formica', 'roundabout', 'drain', 'drying', '11:25', 'westjet', 'us air force', 'comcast', 'soon', 'futon', 'braid', 'us airways', '49', 'red velvet', 'sas', 'cosmo', '100 year party ct', 'in cabbage town']

class _VQA_split1(torch.utils.data.Dataset):
    def __init__(self, root_dir, split, classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        
        
        # NOTE: so tricky
        if root_dir.endswith('vv'):
            root_dir = root_dir[0: -2]
            self.full_label_dim = 700
        else:
            self.full_label_dim = None
        
        self.root_dir = root_dir
        self.data = read_json(os.path.join(root_dir, 'label1.json'))
        
        n = int(len(self.data) * 0.8)
        
        if split == 'train':
            self.data = self.data[: n]
        elif split in ('test', 'val'):
            self.data = self.data[n: ]
            
        # logger.info(f'Loaded {len(self.data)} samples for {split} split')
            
        ignore_classes_idx = [classes.index(c) for c in ignore_classes]
        
        new_self_data = []
        for i, d in enumerate(self.data):
            should_ignore = False
            for label_idx in d[2]:
                if label_idx in ignore_classes_idx:
                    should_ignore = True
                    break
            if not should_ignore:
                # full_label = [0] * (len(classes) - len(ignore_classes))
                new_labels = [idx_map[l] if idx_map is not None else l for l in d[2]]
                # for l in zip(new_labels, d[3]):
                #     full_label[l[0]] = l[1]
                    
                new_self_data.append(
                    (d[0], d[1], new_labels, d[3])
                )
        self.data = new_self_data
        
        logger.info(f'Loaded {len(self.data)} samples for {split} split (after ignoring some classes)')
            
        #self.processor = ViltProcessor.from_pretrained('dandelin/vilt-b32-mlm')
        from transformers import AutoTokenizer,AutoProcessor,GitProcessor
        self.processor = AutoProcessor.from_pretrained('new_impl/mm/Vis_bert/QuestionAnswering/VisBert_pretrained',model_max_length = 30)
        self.classes = classes
        self.ignore_classes = ignore_classes
        self.idx_map = idx_map
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        cv2.setNumThreads(0)
    
        image_id, question, labels, scores = self.data[idx]

        image_path = os.path.join(self.root_dir, f'train2014/COCO_train2014_{str(image_id).zfill(12)}.jpg')
        image = Image.open(image_path).convert('RGB')
        image = Resize((224, 224))(image)
        #encoding = self.processor(image, question, padding='max_length', return_tensors="pt")
        encoding  = self.processor(images = image,text = question,return_tensors = "pt",padding = "max_length")

        # label = self.processor(text="2", return_tensors="pt").input_ids
        # print(label)
        for k in ['input_ids', 'attention_mask', 'pixel_values']:
            encoding[k] = encoding[k][0]  
        label1 = 0
        max = 0
        text = '1'
        full_label = [0] * (len(self.classes) if self.full_label_dim is None else self.full_label_dim)
        for label, score in zip(labels, scores):
            if score > max:
                label1 = self.processor(text = all_classes[label],return_tensors = "pt",padding = 'max_length').input_ids
                max = score
                text = all_classes[label]
            full_label[label] = score
        full_label = torch.FloatTensor(full_label)
        inputs = encoding
        inputs["labels"] = label1
        inputs['labels'] = inputs['labels'][0]
        return encoding, full_label
        #return encoding, inputs , text#这里的这个text就是文本答案

from data.datasets.visual_question_answering.generate_c_image.imagenet_c import corrupt

class _VQA_split1_c(torch.utils.data.Dataset):
    def __init__(self, root_dir, split, corruption_name, classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        
        # NOTE: so tricky
        if root_dir.endswith('vv'):
            root_dir = root_dir[0: -2]
            self.full_label_dim = 700
        else:
            self.full_label_dim = None
            
        self.root_dir = root_dir
        self.data = read_json(os.path.join(root_dir, 'label1.json'))
        self.corruption_name = corruption_name
        
        n = int(len(self.data) * 0.8)
        
        if split == 'train':
            self.data = self.data[: n]
        elif split in ('test', 'val'):
            self.data = self.data[n: ]
            
        logger.info(f'Loaded {len(self.data)} samples for {split} split')
            
        ignore_classes_idx = [classes.index(c) for c in ignore_classes]
        
        new_self_data = []
        for i, d in enumerate(self.data):
            should_ignore = False
            for label_idx in d[2]:
                if label_idx in ignore_classes_idx:
                    should_ignore = True
                    break
            if not should_ignore:
                # full_label = [0] * (len(classes) - len(ignore_classes))
                new_labels = [idx_map[l] if idx_map is not None else l for l in d[2]]
                # for l in zip(new_labels, d[3]):
                #     full_label[l[0]] = l[1]
                    
                new_self_data.append(
                    (d[0], d[1], new_labels, d[3])
                )
        self.data = new_self_data
        
        logger.info(f'Loaded {len(self.data)} samples for {split} split (after ignoring some classes)')
            
        #self.processor = ViltProcessor.from_pretrained('dandelin/vilt-b32-mlm')
        from transformers import AutoTokenizer,AutoProcessor
        self.processor = AutoProcessor.from_pretrained('new_impl/mm/Vis_bert/QuestionAnswering/VisBert_pretrained',model_max_length = 30)

        self.classes = classes
        self.ignore_classes = ignore_classes
        self.idx_map = idx_map
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        cv2.setNumThreads(0)
    
        image_id, question, labels, scores = self.data[idx]

        image_path = os.path.join(self.root_dir, f'train2014/COCO_train2014_{str(image_id).zfill(12)}.jpg')
        image = Image.open(image_path).convert('RGB')
        image = Resize((224, 224))(image)
        
        # key
        image = Image.fromarray(corrupt(np.array(image), severity=5, corruption_name=self.corruption_name))
        
        #encoding = self.processor(image, question, padding='max_length', return_tensors="pt")
        encoding  = self.processor(images = image, text = question,return_tensors = "pt",padding = "max_length")

        for k in ['input_ids', 'attention_mask', 'pixel_values']:
             encoding[k] = encoding[k][0]  
        label1 = 0
        max = 0
        text = '1'
        full_label = [0] * (len(self.classes) if self.full_label_dim is None else self.full_label_dim)
        for label, score in zip(labels, scores):
            if score > max:
                labels = self.processor(text = all_classes[label],return_tensors = "pt",padding = "max_length").input_ids
                max = score
                text = all_classes[label]
            full_label[label] = score
        full_label = torch.FloatTensor(full_label)
        inputs = encoding
        inputs["labels"] = label1
        inputs['labels'] = inputs['labels'][0]
        return encoding, full_label
        #return encoding , inputs , text


class _VQAv2_split1(torch.utils.data.Dataset):
    def __init__(self, root_dir, split, classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        
        
        # NOTE: so tricky
        if root_dir.endswith('vv'):
            root_dir = root_dir[0: -2]
            self.full_label_dim = 700
        else:
            self.full_label_dim = None
        
        self.root_dir = root_dir
        self.data = read_json(os.path.join(root_dir, 'label1.json'))
        
        n = int(len(self.data) * 0.8)
        
        if split == 'train':
            self.data = self.data[: n]
        elif split in ('test', 'val'):
            self.data = self.data[n: ]
            
        # logger.info(f'Loaded {len(self.data)} samples for {split} split')
            
        ignore_classes_idx = [classes.index(c) for c in ignore_classes]
        
        new_self_data = []
        for i, d in enumerate(self.data):
            should_ignore = False
            for label_idx in d[2]:
                if label_idx in ignore_classes_idx:
                    should_ignore = True
                    break
            if not should_ignore:
                # full_label = [0] * (len(classes) - len(ignore_classes))
                new_labels = [idx_map[l] if idx_map is not None else l for l in d[2]]
                # for l in zip(new_labels, d[3]):
                #     full_label[l[0]] = l[1]
                    
                new_self_data.append(
                    (d[0], d[1], new_labels, d[3])
                )
        self.data = new_self_data
        
        logger.info(f'Loaded {len(self.data)} samples for {split} split (after ignoring some classes)')
            
        self.processor = ViltProcessor.from_pretrained('dandelin/vilt-b32-mlm')
        #self.processor = ViltProcessor.from_pretrained('new_impl/mm/Vis_bert/QuestionAnswering/vilt',model_max_length = 40)
        self.classes = classes
        self.ignore_classes = ignore_classes
        self.idx_map = idx_map
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        cv2.setNumThreads(0)
    
        image_id, question, labels, scores = self.data[idx]

        image_path = os.path.join(self.root_dir, f'train2014/COCO_train2014_{str(image_id).zfill(12)}.jpg')
        image = Image.open(image_path).convert('RGB')
        image = Resize((224, 224))(image)
        encoding = self.processor(image, question, padding='max_length', return_tensors="pt")
        for k in ['input_ids', 'token_type_ids', 'attention_mask', 'pixel_values', 'pixel_mask']:
            encoding[k] = encoding[k][0]
            
        full_label = [0] * (len(self.classes) if self.full_label_dim is None else self.full_label_dim)
        for label, score in zip(labels, scores):
            full_label[label] = score
        full_label = torch.FloatTensor(full_label)

        return encoding, full_label
    

from data.datasets.visual_question_answering.generate_c_image.imagenet_c import corrupt

class _VQAv2_split1_c(torch.utils.data.Dataset):
    def __init__(self, root_dir, split, corruption_name, classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        
        # NOTE: so tricky
        if root_dir.endswith('vv'):
            root_dir = root_dir[0: -2]
            self.full_label_dim = 700
        else:
            self.full_label_dim = None
            
        self.root_dir = root_dir
        self.data = read_json(os.path.join(root_dir, 'label1.json'))
        self.corruption_name = corruption_name
        
        n = int(len(self.data) * 0.8)
        
        if split == 'train':
            self.data = self.data[: n]
        elif split in ('test', 'val'):
            self.data = self.data[n: ]
            
        logger.info(f'Loaded {len(self.data)} samples for {split} split')
            
        ignore_classes_idx = [classes.index(c) for c in ignore_classes]
        
        new_self_data = []
        for i, d in enumerate(self.data):
            should_ignore = False
            for label_idx in d[2]:
                if label_idx in ignore_classes_idx:
                    should_ignore = True
                    break
            if not should_ignore:
                # full_label = [0] * (len(classes) - len(ignore_classes))
                new_labels = [idx_map[l] if idx_map is not None else l for l in d[2]]
                # for l in zip(new_labels, d[3]):
                #     full_label[l[0]] = l[1]
                    
                new_self_data.append(
                    (d[0], d[1], new_labels, d[3])
                )
        self.data = new_self_data
        
        logger.info(f'Loaded {len(self.data)} samples for {split} split (after ignoring some classes)')
            
        self.processor = ViltProcessor.from_pretrained('dandelin/vilt-b32-mlm')
        #self.processor = ViltProcessor.from_pretrained('new_impl/mm/Vis_bert/QuestionAnswering/vilt',model_max_length = 40)
        self.classes = classes
        self.ignore_classes = ignore_classes
        self.idx_map = idx_map
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        cv2.setNumThreads(0)
    
        image_id, question, labels, scores = self.data[idx]

        image_path = os.path.join(self.root_dir, f'train2014/COCO_train2014_{str(image_id).zfill(12)}.jpg')
        image = Image.open(image_path).convert('RGB')
        image = Resize((224, 224))(image)
        
        # key
        image = Image.fromarray(corrupt(np.array(image), severity=5, corruption_name=self.corruption_name))
        
        encoding = self.processor(image, question, padding='max_length', return_tensors="pt")
        
        for k in ['input_ids', 'token_type_ids', 'attention_mask', 'pixel_values', 'pixel_mask']:
            encoding[k] = encoding[k][0]
            
        full_label = [0] * (len(self.classes) if self.full_label_dim is None else self.full_label_dim)
        for label, score in zip(labels, scores):
            full_label[label] = score
        full_label = torch.FloatTensor(full_label)

        return encoding, full_label
    
    
class _VQAv2_split2(torch.utils.data.Dataset):
    def __init__(self, root_dir, split, classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        
        # NOTE: so tricky
        if root_dir.endswith('vv'):
            root_dir = root_dir[0: -2]
            self.full_label_dim = 700
        else:
            self.full_label_dim = None
            
            
        self.root_dir = root_dir
        self.data = read_json(os.path.join(root_dir, 'label2.json'))
        
        n = int(len(self.data) * 0.8)
        
        if split == 'train':
            self.data = self.data[: n]
        elif split in ('test', 'val'):
            self.data = self.data[n: ]
            
        # logger.info(f'Loaded {len(self.data)} samples for {split} split')
            
        ignore_classes_idx = [classes.index(c) for c in ignore_classes]
        
        new_self_data = []
        for i, d in enumerate(self.data):
            should_ignore = False
            for label_idx in d[2]:
                if label_idx in ignore_classes_idx:
                    should_ignore = True
                    break
            if not should_ignore:
                # print(idx_map)
                # full_label = [0] * (len(classes) - len(ignore_classes))
                new_labels = [idx_map[l] if idx_map is not None else l for l in d[2]]
                # for l in zip(new_labels, d[3]):
                #     full_label[l[0]] = l[1]
                    
                new_self_data.append(
                    (d[0], d[1], new_labels, d[3])
                )
        self.data = new_self_data
        
        logger.info(f'Loaded {len(self.data)} samples for {split} split (after ignoring some classes)')
            
        self.processor = ViltProcessor.from_pretrained('dandelin/vilt-b32-mlm')
        
        self.classes = classes
        self.ignore_classes = ignore_classes
        self.idx_map = idx_map
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        cv2.setNumThreads(0)
    
        image_id, question, labels, scores = self.data[idx]

        image_path = os.path.join(self.root_dir, f'train2014/COCO_train2014_{str(image_id).zfill(12)}.jpg')
        image = Image.open(image_path).convert('RGB')
        image = Resize((224, 224))(image)
        encoding = self.processor(image, question, padding='max_length', return_tensors="pt")
        
        for k in ['input_ids', 'token_type_ids', 'attention_mask', 'pixel_values', 'pixel_mask']:
            encoding[k] = encoding[k][0]
            
        full_label = [0] * (len(self.classes) if self.full_label_dim is None else self.full_label_dim)
        for label, score in zip(labels, scores):
            full_label[label] = score
        full_label = torch.FloatTensor(full_label)

        return encoding, full_label
        


all_classes = ['net', 'pitcher', 'orange', 'yes', 'white', 'skiing', 'red', 'frisbee', 'brushing teeth', 'no', 'black and white', 'skateboard', '1', 'blue', 'green', 'motorcycle', 'gray', '2', 'purse', 'skis', 'poles', 'surfboard', 'dog', 'on', 'office', 'large', 'very big', 'laptop', 'vent', 'computer', 'black', 'bear', '3', 'wii', 'glasses', 'tree', 'eating', 'log', '5', 'left', 'living room', 'pink', 'right', 'railing', 'grass', 'wire', '10 years', 'knife', 'cake', 'banana', 'chef', 'vanilla', '4', 'outdoor', 'mustard', 'bun', 'clouds', 'dock', 'brown', 'silver', 'refrigerator', 'square', 'teddy', 'elm', 'stripes', 'baseball', 'catcher', 'beer', 'bottom', 'north', 'nike', 'yellow and white', 'morning', 'elephant', 'red and white', 'propeller', 'tan', 'wall', 'clock', 'table', '0', 'wood', 'christmas', 'spinach', 'thick', 'bag', 'leaves', 'necklace', '6', 'bathroom', 'shower', 'towel', 'solid', 'referee', 'wilson', 'e', '24', 'hat', 'grazing', 'sheep', '10', 'tag', 'spanish', 'hot dog', 'plate', 'lunch', 'butter', 'peppers', 'onions', 'very', 'pig', 'sweet', 'flowers', 'floral', 'yellow', 'window', '7', 'pizza', 'car', '', 'cargo', 'stairs', 'abstract', 'rug', 'baseball cap', 'texting', 'pole', 'crosswalk', 'nothing', 'urban', 'bus', 'light', 'afternoon', 'boat', 'cheese', 'paper', 'real', 'sun', 'birthday', 'words', 'inside', 'shadows', 'tomato', 'evergreen', '100 feet', 'trees', 'building', 'hay', 'ski pole', 'walking', 'ice', 'laundry', 'pepsi', 'good', '1:50', 'purple', '13', 'africa', 'teddy bears', 'socks', 'giraffe', 'soccer', 'blue and yellow', 'zebras', 'cupcake', 'broccoli', 'parking lot', 'cows', 'herding', 'on table', 'fish', 'nightstand', '50', 'overcast', 'cross', 'toaster oven', 'tile', '11:55', 'red and yellow', 'nowhere', 'hair dryer', 'truck', '11', 'people', 'rectangle', 'hot dogs', 'party', '12:55', 'apron', 'kitchen', 'cooking', 'ring', '1 way', 'stop', 'neither', 'many', 'female', 'brushing', 'tie', 'tennis racket', 'knife and fork', 'restaurant', 'cat', 'bed', 'sand', 'ocean', 'cold', 'kites', 'cumulus', 'standing', 'male', 'star', 'tracks', 'chocolate', 'round', 'fork and knife', 'yankees', 'pictures', 'dots', 'bird', 'parrot', 'red white and blue', 'man', 'metal', 'fence', 'snowboarding', 'pine', 'snow', 'shorts', 'swim', 'wine', 'brick', 'no parking', 'children', 'beef', 'phone', 'english', 'cell phone', 'pink and yellow', 'clear', 'watermelon', 'bedroom', 'fork', 'cow', 'rackets', 'tennis rackets', '8', 'collar', 'tennis', 'playing tennis', 'skirt', '30', 'polka dot', 'beach', 'horse', 'grill', 'african american', 'down', 'street', 'in air', 'sweater', 'yellow and blue', 'park', 'spectators', 'parasailing', '31', 'river', '55', 'shadow', 'winter', 'chicken', 'tea', 'evening', 'dusk', 'ski resort', 'helmet', 'bench', 'resting', 'elephants', 'southwest', 'usa', 'cars', 'town', 'bananas', 'umbrella', 'container', 'woman', 'on counter', 'salad', 'striped', 'motel', 'vertical', 'oranges', 'hot sauce', 'bottle', 'juice', 'eyes', 'ground', 'backpack', 'black and yellow', 'forward', 'jackets', '1 on right', 'green and yellow', 'playing baseball', 'riding', 'sitting', 'carrot', 'basket', 'seagull', 'ski poles', 'p', 'parking', 'street light', 'strap', 'bike', 'riding bike', 'poodle', 'shoes', 'carpet', 'lettuce', 'food', '1 foot', 'roses', 'mountains', 'scissors', 'camera', 'beige', 'beard', 'cutting', 'baby', 'tape', 'watch', 'never', 'taking picture', 'eggs', 'syrup', 'sandwich', 'water skiing', 'microphone', 'back', 'bears', 'donuts', 'w', 'sky', 'double decker', 'england', 'surfing', 'running', 'shirt', 'barn', 'weather vane', 'white and blue', 'fishing', 'bridge', 'los angeles', 'open', 'red sox', 'bat', 'plane', 'white and green', 'transportation', 'sunny', 'bus stop', 'city', 'brown and white', 'bicycle', 'crow', 'magazines', 'daisy', '14', 'old', 'curtains', 'snowboard', 'dinosaur', 'racing', 'asphalt', 'court', 'plastic', 'circle', 'red and blue', 'zebra', '12', 'biplane', 'shallow', 'brazil', 'logo', '2:20', 'electric', 'motion', 'toothbrushes', 'orange and white', '66', 'spoon', 'toyota', 'tennis shoes', '46', 'second', 'no 1', 'iphone', 'friend', 'apple', '15', 'tiger', 'glove', 'airplane', 'bow', 'air france', 'passengers', 'tv', 'on building', '3:55', 'victorian', 'steeple', 'happy', 'skateboarding', 'fruit', 'cutting board', 'cantaloupe', 'kiwi', 'sliced', 'heart', 'water', 'rainy', 'carrots', 'giraffes', 'eat', 'ramp', 'lab', 'field', 'horizontal', 'birds', 'home', 'shrimp', '12 feet', 'girl', 'modern', 'dell', 'boots', 'sunglasses', 'black and orange', 'yellow and black', 'gloves', 'hp', 'desk', 'both', 'sign', 'on street', '2000', 'cirrus', 'ceiling', 'fluorescent', 'up', '9', 'boys', 'playing soccer', 'american', 'passenger', 'turn', 'palm', 'wedding', 'branch', 'parrots', 'air force', 'on tracks', 'small', 'dirty', 'france', 'honda', '2.00', 'vase', 'flying', 'driving', 'tissue', 'protest', 'corona', 'twin', 'clothes', 't shirt', 'window sill', 'wild', 'noon', 'caution', 'spring', 'raining', 'cane', 'school', 'windsurfing', 'parachute', 'black and red', '25', 'background', 'toaster', 'planes', 'yellow and red', 'spatula', '10:10', 'ivory', 'train', 'highway', 'off', 'on track', 'electricity', 'italy', 'dinner', 'sink', 'squares', '5 ft', 'parked', 'store', 'dress', 'signs', 'football', 'rugby', 'stainless steel', 'dirt', 'blue and white', 'klm', 'house', 'unknown', 'ford', 'reading', 'chair', 'mountain', 'alive', 'water skis', 'picture', 'parade', 'trailer', 'boating', 'holding it', 'shade', 'cloth', 'candle', 'hose', 'hand', '3:25', 'on sidewalk', 'poster', 'downhill', 'reflection', 'summer', 'pickles', 'halloween', 'bats', 'london', 'zoo', 'surfer', 'racket', 'flickr', 'cutting hair', 'strawberries', 'mushroom', 'teddy bear', 'big', 'suitcase', 'veggie', 'pepper', 'houses', '70', 'toshiba', 'triangle', 'boxes', 'photograph', 'smoke', 'engine', 'camel', 'sidewalk', 'left 1', 'red and green', '4:35', 'on couch', 'candy', 'homemade', 'mouse', 'box', 'movie', '45', 'strawberry', 'fridge', 'full', 'vegetables', 'bright', 'play', 'remote', 'pond', 'savannah', 'celery', 'concrete', 'semi', 'scania', 'safety', 'posing', 'fabric', 'laying', 'couch', 'blueberries', 'handle', 'pipe', 'stick', 'steak', 'chain link', 'barbed wire', 'mozzarella', 'soda', 'fire hydrant', 'cat food', 'pepperoni', 'lot', 'licking', 'red and black', 'clay', 'tennis court', 'jumping', 'potatoes', 'toothbrush', 'kite', 'flying kite', 'broken', 'black and silver', 'lap', 'outside', '44', 'delta', 'greyhound', 'talking on phone', 'bad', 'kettle', '35', 'motorcycles', 'produce', 'steering wheel', '18', 'humans', 'coffee', 'white and brown', 'fall', 'bread', 'cherry', '4:30', 'flag', 'night', 'lamp', 'cucumber', 'porcelain', 'oval', 'museum', 'rain', 'sprinkles', '20', 'kids', 'bracelet', 'sneakers', 'mask', 'mickey mouse', 'very high', 'costume', 'cabbage', 'paint', 'lighting', 'young', 'air conditioner', 'wooden', 'board', 'beets', '16', 'lights', 'ladder', 'glass', 'fries', 'steamed', 'shepherd', 'cotton', 'suit', 'goatee', 'on his head', 'print', 'happy birthday', 'forks', 'travel', 'maple', '200', 'oil', 'jeans', 'can', 'chopsticks', 'on wall', 'construction', '36', 'chinese', 'festival', 'gas', 'throwing', 'circus', 'wires', 'not possible', 'plates', 'sugar', 'in', "women's", 'door', 'volleyball', 'serving', 'ponytail', 'business', 'decoration', 'santa', 'flat', 'barrel', '12:15', 'candles', 'free', 'hair', 'ball', 'stop sign', 'wetsuit', 'green and black', 'foreground', 'stands', 'china airlines', 'flower', '300', 'on bench', 'plaster', 'phones', 'sailboat', 'apples', 'road', 'recently', 'cones', 'cactus', 'rice', 'vegetarian', 'donut', 'ketchup', 'police', 'mirror', 'rock', 'meat', 'blinds', 'cell phones', 'china', 'rust', '7:25', 'stone', 'vans', 'middle', 'eagle', '9:30', 'ping pong', 'microwave', 'gmc', 'umbrellas', 'wrist', 'laughing', 'boy', 'next to toilet', 'tabby', 'petting', 'south', '40', 'checkered', 'slow', 'cardboard', 'windows', 'croissant', 'plain', 'cookie', 'on ground', 'low', 'water bottle', 'goggles', 'turkey', 'shut', 'kite flying', 'bowl', 'smile', 'in bowl', 'bush', 'cloudy', 'top left', 'skateboarder', 'coca cola', 'pan', 'drinking', 'short', 'floor', 'thanksgiving', 'radio', 'drink', 'on toilet', 'bike rack', 'bleachers', 'train tracks', 'horses', 'far', 'top', 'toilet', 'in water', 'private', 'nature', 'commercial', 'stroller', 'power', 'stuffed animals', 'uniforms', 'japan', 'faucet', 'green and orange', 'corn', 'white and yellow', 'mercedes', 'in sky', 'tarp', 'indian', 'counter', 'multicolored', 'polar', 'go', 'no number', 'swimming', 'bridle', 'cowboy', 'olives', 'pizza cutter', 'british airways', 'nighttime', 'australia', 'tiles', 'pug', 'wicker', 'british', 'us airways express', 'burton', 'christmas tree', 'napkin', 'writing', 'rocks', 'hello kitty', 'gold', 'fan', 'skateboards', 'day', 'on floor', '2008', 'dark', 'flying kites', 'rural', 'olympics', 'bmw', '34', 'denim', 'typing', 'for fun', 'steel', 'watching tv', 'driver', 'grapes', 'f', 'angels', 'roof', 'handlebars', 'train station', 'public', 'oak', 'sleeping', 'canada', 'air canada', 'on top', 'tired', 'blonde', 'cups', 'little', 'adidas', '10 feet', 'white and gray', 'leaf', 'fisheye', 'forest', 'war', 'octagon', 'raspberry', 'helmets', 'united states', '29', 'noodles', 'van', 'long', 'traveling', 'luggage', 'airport', 'single', 'pitching', 'dugout', 'garbage', 'happiness', 'cigarette', 'on tower', 'antelope', 'graffiti', 'skating', 'on road', 'curved', 'washington', 'ski lift', 'athletics', 'brace', 'squatting', 'catching', 'batter', 'batting', 'game', 'towards', '33', 'sliding', 'makeup', 'japanese', 'person', 'pirates', 'plaid', 'rose', 'daytime', 'keyboard', 'surfboards', 'hummingbird', 'ollie', '11:30', 'clock tower', 'san francisco', 'stopping', 'tags', 'samsung', 'computers', 'cabinets', 'talking', 'asparagus', '5 years', 'adult', 'rabbit', 'empty', 'softball', '1st', 'playing', 'chairs', 'farm', 'cross country', 'dump truck', 'women', 'snowboarder', 'tall', 'monkey', 'fire', 'books', 'cessna', 'chandelier', 'dunkin donuts', 'beans', 'relish', 'parking meter', 'ducks', 'sandals', 'doughnut', 'lighthouse', 'yacht', 'german shepherd', 'raw', 'chain', '2 feet', 'pedestal', 'mutt', 'race', 'poor', 'cat and dog', 'station', 'printer', 'daisies', 'front', 'gravel', 'grassy', 'pigeons', 'dogs', 'in car', 'life', 'wii remotes', 'suv', 'leather', 'bottom right', 'peace', 'blanket', 'frisbees', '12:30', 'scooter', 'going', 'analog', 'america', 'pitbull', 'relaxing', 'paddle boarding', 'white and pink', 'ride', 'side', 'on desk', 'on chair', '2012', 'multi', 'straight', 'big ben', 'closed', '3 feet', 'waves', 'buoy', 'trash can', 'medium', 'very tall', 'yamaha', 'sunlight', 'hit ball', 'dry', 'coke', 'gym', 'orange and black', 'center', 'rope', 'flip flops', 'siamese', 'crafts', 'color', 'italian', 'playing frisbee', 'skate park', 'orange juice', 'windowsill', 'thumb', 'pie', 'toast', 'no hat', 'benches', 'diamond', 'blender', 'avocado', 'television', 'speakers', 'pony', 'baseball field', 'pavement', 'not there', 'diamonds', '4 feet', 'goalie', 'soccer ball', 'runway', 'video game', 'gaming', 'casual', 'green and white', 'toilet brush', 'working', 'pickup', 'girls', 'remotes', 'pasta', 'hood', 'braves', 'skier', 'motorola', '17', 'b', '100', 'hospital', 'wagon', 'milk', 'ferry', 'rainbow', 'on bed', 'toward', '1:30', '19', 'mercedes benz', 'supreme', 'thin', 'platform', 'thai', 'storage', 'swan', 'peach', '10:05', 'dome', 'chiquita', '2:00', 'mountain dew', '23', 'knives', 'street sign', 'on beach', 'playing wii', 'stickers', 'yogurt', 'on grass', '9:45', 'gatorade', 'umpire', '37', 'desktop', 'desserts', 'main', 'boston', 'fell', 'top right', 'case', 'asleep', 'over', 'grapefruit', 'breakfast', 'headphones', 'freight', 'cup', 'sweatband', 'nobody', 'lamps', '9:25', 'scarf', 'on fridge', 'moving', 'fresh', 'blue jay', 'chihuahua', 'ceramic', 'mushrooms', 'on plate', 'human', 'power lines', 'hotel', 'map', 'earring', 'boarding', 'warm', 'napkins', 'brown and black', 'broom', 'basketball', 'papers', 'sad', 'kickstand', '60', 'shoulder', 'sleep', 'footprints', 'tunnel', '1990', 'hats', '6 inches', 'ham', 'bacon', 'church', '53', 'pineapple', 'at camera', 'red bull', 'pilot', 'tattoo', 'work', 'polar bear', 'taking off', 'website', '22', '4:00', 'coffee maker', 'fast', 'fur', 'rubber', 'tongs', 'german', 'germany', 'toy', '3:20', 'calm', 'pots', 'fruits', '9:20', 'drawer', 'oven', 'soup', 'stove', 'heels', 'wind', 'island', 'blood', 'leg', 'theater', 'tennis racquet', '21', 'gothic', '2:35', 'wii remote', 'turning', '20 feet', 'ears', 'fun', 'to right', 'child', 'fly', 'head', 'drywall', 'pier', 'feeding giraffe', 'in vase', 'burger', 'easter', 'onion', 'uniform', 'guitar', 'time', 'tomatoes', 'ship', 'tulips', 'glaze', 'tent', 'market', 'bandana', 'still', "don't know", 'piano', 'mouth', 'run', 'sparrow', 'lines', 'vest', '1950', 'jet', 'sepia', '2015', 'busy', 'dessert', '75', 'finch', 'pastries', 'outdoors', 'bakery', 'clean', 'ipod', 'tablecloth', 'looking at phone', 'in front', 'food truck', 'face', 'swinging', 'safari', '500', 'volkswagen', '2010', 'shelves', 'riding horses', '2016', 'towels', 'lemon', 'straw', 'bamboo', '5 feet', 'hardwood', 'h', 'meter', 'charging', 'bald', 'caucasian', 'man on left', 'stand', '27', 'dining room', 'sandwiches', '32', 'apartment', 'tower', 'virgin', 'out', 'white and red', "i don't know", 'chains', 'legs', 'goats', 's', 'dresser', 'camper', 'half', 'decorative', 'hawaiian', 'wheel', 'florida', 'reds', 'washington dc', 'moon', 'conference', 'screen', 'controller', 'robin', 'men', 'protection', 'harley davidson', 'coal', 'mustache', 'smiling', 'pedestrians', 'me', 'tray', 'monitor', 'bell', 'landscape', 'club', 'toothpick', 'seagulls', 'bowtie', 'lake', 'steam', 'surf', 'baseball glove', 'blinders', 'woods', 'shearing', 'dad', 'mixer', 'pot', 'blending', 'identification', 'owl', 'wine glass', 'new york', 'yarn', 'tennis ball', 'ice cream', 'chevrolet', 'shirt and tie', 'taking selfie', 'blue and green', "he isn't", 'cutting cake', 'east', 'setting', '7 eleven', 'stars', 'jockey', 'jacket', 'book', 'gray and white', 'pen', 'red white blue', 'above', 'alaska', 'tongue', 'feathers', 'k', 'camping', 'corner', 'away', 'ski', 'texas', 'fire truck', 'sailboats', 'jump', 'walk', 'spray paint', 'loading', 'united', '1000', 'roman numerals', 'surprise', '3rd', 'first', 'side of road', 'dodgers', 'airplanes', 'unsure', 'russian', 'wet', '5 star', 'blankets', 'natural', 'across street', 'smartphone', 'duck', 'sausage', 'paris', 'newspaper', 'pants', 'spices', 'pillow', 'to left', 'snowboards', 'colgate', 'on elephant', 'string', 'horns', '2:40', "men's", 'cobblestone', 'regular', 'staring', '28', 'barber shop', 'cut', 'x', 'above sink', 'above stove', 'dishes', 'dalmatian', 'watching', 'glazed', '5:25', 'messy', 'wallet', 'tuna', 'grilled', 'french', 'green and blue', 'sunflowers', 'wool', 'cabinet', 'shell', 'foil', 'bottles', 'bar', 'king', 'paper towels', 'friends', 'beagle', 'school bus', 'laptops', 'snowing', 'cement', 'pc', 'accident', 'stuffed animal', 'balance', 'white and black', 'cleats', 'on sink', 'pool', 'mom', 'downtown', 'asian', 'heater', 'bathing', '193', 'against wall', 'canopy', 'berries', 'military', 'pickle', 'clams', 'seafood', 'in box', 'boats', 'lizard', 'lemonade', 'm', 'soft', 'country', 'for sale', 'arm', 'listening', 'curly', 'play tennis', 'hands', 'cereal', 'blue and red', 'robe', 'soap', 'trains', 'throwing frisbee', 'smoking', 'india', 'headband', 'not very', 'westin', 'serve', 'bicycles', "can't tell", 'visibility', 'ana', 'reins', 'rodeo', 'riding motorcycle', 'mexico', 'mother', 'african', 'left and right', 'button', 'earrings', 'blackberry', 'cell', '10:00', 'harness', 'pillows', 'vegetable', 'tablet', 'fern', 'cats', 'golden retriever', 'goat', 'tractor', "valentine's day", 'hearts', 'khaki', 'man on right', "mcdonald's", 'arriving', 'husky', 'on skateboard', 'vases', 'coat', 'beanie', 'coming', 'granite', 'sports', 'leash', 'balls', 'blurry', 'baseball bat', 'mug', 'eiffel tower', 'worms', 'trash', 'terrier', 'painting', 'rooster', '42', 'jones', 'state farm', 'balloon', 'trunk', 'coach', 't', 'playing game', 'fireplace', 'behind clouds', 'uphill', 'motocross', 'sony', 'magazine', 'kitesurfing', 'catching frisbee', 'catch frisbee', 'bud light', 'fighting', '1 on left', 'very old', 'hallway', 'lexus', 'wii controller', '5:45', 'catholic', 'muffin', 'traffic light', 'grocery', 'shelf', '2:25', 'honey', 'plants', 'oars', 'foggy', "nathan's", 'cord', 'yard', '48', 'chimney', 'calico', 'suits', 'sideways', 'animals', 'black and blue', 'bikini', 'photographer', 'queen', '1:00', '12:05', 'horseback riding', 'awake', 'bunny', '12:00', 'continental', 'rye', 'family', 'lots', 'owner', 'palm tree', 'design', 'far right', 'tire', 'younger', 'biking', 'giants', 'caramel', 'polo', 'emirates', 'magnets', 'mat', 'ivy', 'cakes', 'bob', 'asia', 'graduation', 'cauliflower', 'c', 'rough', 'air', 'windy', 'victoria', 'trick', 'labrador', 'on left', 'yellow and green', 'butterfly', 'fake', 'on napkin', 'bricks', 'wine glasses', 'detroit', "man's", 'parsley', 'art', 'subway', 'wave', 'placemat', 'hydrant', 'sofa', 'pigeon', 'all', 'branches', 'plant', 'to eat', 'zucchini', 'feta', 'mouse pad', 'cloud', 'toilet paper', 'pumpkin', 'rowing', 'handicap', 'seeds', 'fly kite', 'chicago', 'marble', 'frame', '150', 'rocky', 'sauce', "it's not", 'control', 'high chair', 'playstation', 'xbox', 'roman', 'land', '1:35', 'lifeguard', 'size', 'bull', 'goose', '8 feet', 'recessed', 'statue', 'index', 'phillies', 'strike', 'mirrors', 'pointing', 'farmer', 'collie', 'motorbike', 'lanes', 'bikes', 'gas station', 'logs', 'smaller', 'desert', 'yield', 'flags', 'stool', 'kitten', 'doll', 'daffodils', 'letters', 'dishwasher', 'nuts', '2013', 'persian', 'swim trunks', 'deep', 'doubles', 'in field', 'wristband', 'wheels', 'baking', '4:15', '11:00', 'ear', '2007', '51', 'frog', 'boogie board', 'hungry', 'by window', 'ambulance', 'pigtails', 'microsoft', 'on man', 'laying down', '3:00', 'taxi', 'pedestrian', 'landing', 'numbers', '38', 'stones', 'clocks', 'new', 'picnic', 'fog', 'buffalo', 'under armour', 'orioles', 'bags', 'golden gate', 'castle', 'canoe', 'selfie', 'cream', 'floating', 'indoor', 'antique', 'aluminum', 'peas', 'sun hat', 'on right', 'flour', 'under sink', 'fashion', 'fedora', 'shells', '1 hour', 'puppy', 'motor', '120', 'sail', 'mexican', 'dead end', 'paddle', 'shop', 'boxing', 'birthday cake', 'chalk', 'style', 'nissan', 'sticker', 'north face', 'squash', 'not sure', 'seat', 'himself', 'circles', 'san diego', 'kia', 'mattress', 'obama', 'lamb', 'american flag', 'climbing', 'skull and crossbones', 'roast beef', 'visor', 'double', '52', 'high', 'stagecoach', 'cart', 'feeding', 'eaten', 'cone', 'smoothie', 'golf', 'colorado', 'electronics', '5:15', 'bowling', 'players', 'ketchup and mustard', 'styrofoam', '6 feet', 'hawk', 'cheddar', 'arabic', 'shower curtain', 'army', 'salmon', 'hanging', 'whole', 'behind fence', 'bars', 'moss', 'no dog', 'traffic', 'r', 'countryside', 'directions', 'cooked', 'aa', '6:45', '4 way', 'stripe', 'brand', 'baseball player', 'bunk', 'coleslaw', 'europe', 'dead', 'arch', 'scrambled', 'clothing', 'closet', 'egg', 'suitcases', 'indoors', 'tires', 'lilies', 'cafe', 'toothpaste', 'in background', 'tarmac', 'painted', 'sunset', 'orange and yellow', 'zebra and giraffe', 'ladybug', 'hills', 'tail', 'couple', 'kawasaki', 'smooth', 'powdered sugar', 'pedestrian crossing', 'french fries', 'teeth', 'ribbon', 'saddle', 'on train', '39', 'curb', 'tow', 'shark', 'white and orange', 'gravy', 'curtain', 'lime', 'skull', 'crossing', 'peacock', 'neck', 'hit', 'dragon', 'tissues', 'basil', 'waving', 'helicopter', 'mud', 'us', 'red and gray', 'sunflower', 'wallpaper', '11:20', 'seattle', 'bookshelf', 'looking', '1 inch', 'harley', 'urinal', 'navy', 'fedex', 'rays', 'deck', 'coaster', '1:20', '4:20', '5:00', 'jp morgan', 'palm trees', 'tub', 'pens', '2 people', 'speaker', 'hamburger', 'green beans', "it isn't", '10:20', 'buildings', 'on shelf', 'orange and blue', '90', 'north america', 'arrow', 'news', 'tropicana', 'formal', 'in grass', 'thumbs up', 'clip', 'tennis player', 'pastry', 'nose', 'pacifier', '11:35', 'different teams', 'cardinals', 'bagel', 'huge', 'out of focus', 'cook', 'wheat', 'photo', 'sedan', 'lanyard', 'pink and white', 'sesame', 'space', 'warning', 'snowy', 'tater tots', 'tropical', 'grandfather', 'mac', 'pajamas', '350', 'casserole', 'pelican', '2009', 'clydesdale', 'tow truck', 'belt', 'west', 'omelet', 'heavy', 'crown', 'in corner', 'hexagon', 'mound', 'iris', 'g', '2:15', '3:10', 'drawing', 'only', 'washing', 'nokia', 'windsor', 'icing', 'several', 'no smoking', 'kayak', 'frosting', 'jetblue', 'shoe', 'britain', 'ties', 'bank', 'camouflage', 'privacy', 'bib', 'blue and gray', 'looking out window', 'falling', 'bucket', 'cupcakes', 'throw ball', 'garden', 'almonds', 'starbucks', 'all way', 'home plate', 'base', 'toys', '1 in front', 'foot', 'california', 'towing', 'cheesecake', 'bushes', 'bow tie', 'down street', '2011', 'police officer', 'windmill', 'taking pictures', 'cleaning', 'on pole', 'main street', 'catch ball', 'mario', 'track', 'garage', "they aren't", 'tents', 'tattoos', '2:45', 'wheelchair', 'money', 'top hat', 'willow', 'brushing hair', '80', 'green and red', 'barrier', 'hiking', 'tank top', 'lufthansa', 'menu', 'forehand', 'wii controllers', 'hundreds', 'water ski', 'furniture', 'paisley', 'pizza hut', 'hill', 'prom', 'tiara', 'students', 'information', 'hazy', 'canon', 'bird feeder', 'crane', 'dr pepper', 'logitech', '2:10', 'all of them', 'utensils', 'telephone', 'converse', 'bone', 'jeep', 'nursing', 'krispy kreme', 'ranch', 'polka dots', 'railroad crossing', 'shirts', 'feeder', 'above toilet', 'unclear', 'below', '43', 'spoons', 'calendar', 'mint', 'spiderman', 'lg', 'concert', 'coats', 'lady', 'dodge', 'flat screen', '10:30', 'music', 'polar bears', 'riding horse', 'cookies', 'hot', 'behind', 'dole', '26', 'pans', 'love', 'winnie pooh', 'copyright', '2 hours', 'snowsuit', 'kissing', 'backhand', 'swans', 'nintendo', 'direction', 'waiting', 'mohawk', 'rail', 'hoodie', 'feet', '106', '10:55', 'coins', 'mitt', 'room', 'adults', 'cameras', 'marker', 'sled', 'conductor', 'farmers market', 'toiletries', 'blue and black', 'sprite', 'bank of america', 'heat', 'emergency', 'hard', '41', '6:00', 'in his hand', 'cluttered', 'grizzly', 'not', 'in hand', 'under table', 'd', 'hitting ball', 'photography', 'intersection', 'backwards', 'crocs', 'chips', 'harry potter', 'hawaii', 'half full', 'carriage', 'curious', 'geese', 'pork', 'l', 'sidecar', 'penguin', 'to see', 'pocket', 'steps', 'cubs', 'junk', 'deer', 'ottoman', 'salt', 'condiments', '1:55', 'post', 'bulldog', 'notebook', 'no cat', 'jets', 'knee pads', 'throw frisbee', 'drinks', 'leopard', 'grape', 'wine tasting', 'baskets', 'santa hat', 'chest', 'sewing', 'on car', 'sony ericsson', 'peeing', 'tour', 'fire extinguisher', 'lemons', 'wiimote', 'guitar hero', 'stopped', 'library', 'blue and pink', 'choppy', 'sailing', 'brush', 'jelly', 'dairy queen', 'shaking hands', 'ge', 'tigers', 'tokyo', 'buses', 'pink and blue', 'singles', 'iron', "don't walk", 'classroom', 'harbor', 'residential', 'joshua', 'uk', 'burgers', 'lace', 'overalls', 'ram', 'dancing', '47', 'shed', 'lid', "he's not", 'amtrak', 'ostrich', 'bathtub', '2:50', 'mall', 'slow down', 'hammer time', 'octopus', 'crib', 'broadway', 'pottery', 'wavy', 'holding phone', 'tusks', 'dining', 'packing', 'thomas', 'budweiser', 'beijing', '11:10', 'wide', 'slope', 'black and gray', 'chili', 'siblings', 'kayaking', 'captivity', 'rack', 'panda', 'pelicans', 'genetics', 'not in service', 'v', 'on laptop', 'gone', 'tying tie', 'scale', 'lily', 'cool', 'n', 'toilets', 'tree branch', 'copper', '870', 'shopping', 'batman', 'black and brown', 'legos', 'drinking water', 'burrito', 'spiral', 'ibm', 'tools', 'cherries', 'maple leaf', 'vines', 'sushi', 'baker', 'globe', 'wireless', 'compaq', 'do not enter', '1:05', 'advertisement', 'movement', 'model', 'hammock', 'swing', 'sheet', 'google', 'right 1', 'haircut', 'exit', 'tim hortons', 'lego', 'cucumbers', 'potato', 'egg salad', 'controllers', 'upside down', 'lion', 'camo', 'dirt bike', 'playing video games', 'crates', 'horizontally', 'plunger', 'radiator', 'in basket', 'cap', 'living', 'briefcase', 'ascending', 'flip phone', '101', 'gun', 'foam', 'serious', 'pancakes', 'heineken', 'driveway', 'cleaner', 'delivery', 'commuter', 'apple and banana', 'chase', 'trucks', 'trunks', '64', 'slacks', 'skiers', 'carrot cake', 'holding', 'surfers', 'horse racing', 'orchid', 'leaving', 'pitch', 'crest', 'miami', 'bus station', 'take off', 'diesel', 'pm', 'wetsuits', '7:35', 'tie dye', 'baked', 'life jacket', 'grilled cheese', 'meatballs', 'monster', 'smiley face', 'keys', 'straight ahead', 'badminton', 'end', '5:05', '10:50', 'each other', 'weeds', 'tinkerbell', 'rottweiler', 'apartments', 'sweatshirt', 'shore', 'switzerland', '65', 'jar', 'skate', 'raspberries', 'singing', 'on bus', 'carnations', 'descending', 'hsbc', 'space needle', 'skatepark', 'kenmore', 'db', "baby's breath", 'shelter', '1980', 'no left turn', '9:05', 'pipes', 'donkey', 'mitsubishi', 'tell time', 'outfield', 'flip', 'stadium', 'heinz', 'distance', 'macaroni', 'on plane', 'triumph', '4:50', 'on stove', 'shih tzu', 'fried', 'sunrise', '2nd', 'suzuki', 'traffic lights', 'hitting', 'healthy', 'tulip', 'right side', 'on sign', 'maroon', '5:40', 'michigan', 'close', 'license plate', 'sniffing', '1:15', 'cardinal', 'older', 'nest', 'colored', 'in back', 'formica', 'roundabout', 'drain', 'drying', '11:25', 'westjet', 'us air force', 'comcast', 'soon', 'futon', 'braid', 'us airways', '49', 'red velvet', 'sas', 'cosmo', '100 year party ct', 'in cabbage town']

@dataset_register(
    name='VQA_split1', 
    classes=all_classes[0: 100], 
    task_type='Visual Question Answering',
    object_type='Generic Object',
    class_aliases=[],
    shift_type=None
)
class VQAv2_split1(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose], 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        if transform is None:
            transform = None
            self.transform = transform
        dataset = _VQA_split1(root_dir, split, classes, ignore_classes, idx_map)
        return dataset

@dataset_register(
    name='VQA_split1_c', 
    classes=all_classes[0: 100], 
    task_type='Visual Question Answering',
    object_type='Generic Object',
    class_aliases=[],
    shift_type=None
)
class VQAv2_split1_c(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose], 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        
        if transform is None:
            transform = None
            self.transform = transform
        dataset = _VQA_split1_c(root_dir, split, "gaussian_noise",classes, ignore_classes, idx_map)
        return dataset


@dataset_register(
    name='VQAv2_split1', 
    classes=all_classes[0: 100], 
    task_type='Visual Question Answering',
    object_type='Generic Object',
    class_aliases=[],
    shift_type=None
)
class VQAv2_split1(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose], 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        if transform is None:
            transform = None
            self.transform = transform
        dataset = _VQAv2_split1(root_dir, split, classes, ignore_classes, idx_map)
        return dataset
    
    
@dataset_register(
    name='VQAv2_split1_c', 
    classes=all_classes[0: 100], 
    task_type='Visual Question Answering',
    object_type='Generic Object',
    class_aliases=[],
    shift_type=None
)
class VQAv2_split1_c(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose], 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        
        if transform is None:
            transform = None
            self.transform = transform
        dataset = _VQAv2_split1_c(root_dir, split, classes, ignore_classes, idx_map)
        return dataset
    
    
@dataset_register(
    name='VQAv2_split2', 
    classes=all_classes[100: ], 
    task_type='Visual Question Answering',
    object_type='Generic Object',
    class_aliases=[],
    shift_type=None
)
class VQAv2_split2(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose], 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        
        print(len(all_classes), len(ignore_classes))
        
        if transform is None:
            transform = None
            self.transform = transform
        dataset = _VQAv2_split2(root_dir, split, classes, ignore_classes, idx_map)
        return dataset
    
    
    

@dataset_register(
    name='VQAv2_split1_c_gaussian_noise', 
    classes=all_classes[0: 100], 
    task_type='Visual Question Answering',
    object_type='Generic Object',
    class_aliases=[],
    shift_type=None
)
class VQAv2_split1_c_gaussian_noise(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose], 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        
        if transform is None:
            transform = None
            self.transform = transform
        dataset = _VQAv2_split1_c(root_dir, split, "gaussian_noise", classes, ignore_classes, idx_map)
        return dataset



@dataset_register(
    name='VQAv2_split1_c_shot_noise', 
    classes=all_classes[0: 100], 
    task_type='Visual Question Answering',
    object_type='Generic Object',
    class_aliases=[],
    shift_type=None
)
class VQAv2_split1_c_shot_noise(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose], 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        
        if transform is None:
            transform = None
            self.transform = transform
        dataset = _VQAv2_split1_c(root_dir, split, "shot_noise", classes, ignore_classes, idx_map)
        return dataset



@dataset_register(
    name='VQAv2_split1_c_impulse_noise', 
    classes=all_classes[0: 100], 
    task_type='Visual Question Answering',
    object_type='Generic Object',
    class_aliases=[],
    shift_type=None
)
class VQAv2_split1_c_impulse_noise(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose], 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        
        if transform is None:
            transform = None
            self.transform = transform
        dataset = _VQAv2_split1_c(root_dir, split, "impulse_noise", classes, ignore_classes, idx_map)
        return dataset



@dataset_register(
    name='VQAv2_split1_c_defocus_blur', 
    classes=all_classes[0: 100], 
    task_type='Visual Question Answering',
    object_type='Generic Object',
    class_aliases=[],
    shift_type=None
)
class VQAv2_split1_c_defocus_blur(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose], 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        
        if transform is None:
            transform = None
            self.transform = transform
        dataset = _VQAv2_split1_c(root_dir, split, "defocus_blur", classes, ignore_classes, idx_map)
        return dataset



@dataset_register(
    name='VQAv2_split1_c_glass_blur', 
    classes=all_classes[0: 100], 
    task_type='Visual Question Answering',
    object_type='Generic Object',
    class_aliases=[],
    shift_type=None
)
class VQAv2_split1_c_glass_blur(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose], 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        
        if transform is None:
            transform = None
            self.transform = transform
        dataset = _VQAv2_split1_c(root_dir, split, "glass_blur", classes, ignore_classes, idx_map)
        return dataset



@dataset_register(
    name='VQAv2_split1_c_motion_blur', 
    classes=all_classes[0: 100], 
    task_type='Visual Question Answering',
    object_type='Generic Object',
    class_aliases=[],
    shift_type=None
)
class VQAv2_split1_c_motion_blur(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose], 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        
        if transform is None:
            transform = None
            self.transform = transform
        dataset = _VQAv2_split1_c(root_dir, split, "motion_blur", classes, ignore_classes, idx_map)
        return dataset



@dataset_register(
    name='VQAv2_split1_c_zoom_blur', 
    classes=all_classes[0: 100], 
    task_type='Visual Question Answering',
    object_type='Generic Object',
    class_aliases=[],
    shift_type=None
)
class VQAv2_split1_c_zoom_blur(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose], 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        
        if transform is None:
            transform = None
            self.transform = transform
        dataset = _VQAv2_split1_c(root_dir, split, "zoom_blur", classes, ignore_classes, idx_map)
        return dataset



@dataset_register(
    name='VQAv2_split1_c_snow', 
    classes=all_classes[0: 100], 
    task_type='Visual Question Answering',
    object_type='Generic Object',
    class_aliases=[],
    shift_type=None
)
class VQAv2_split1_c_snow(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose], 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        
        if transform is None:
            transform = None
            self.transform = transform
        dataset = _VQAv2_split1_c(root_dir, split, "snow", classes, ignore_classes, idx_map)
        return dataset



@dataset_register(
    name='VQAv2_split1_c_frost', 
    classes=all_classes[0: 100], 
    task_type='Visual Question Answering',
    object_type='Generic Object',
    class_aliases=[],
    shift_type=None
)
class VQAv2_split1_c_frost(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose], 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        
        if transform is None:
            transform = None
            self.transform = transform
        dataset = _VQAv2_split1_c(root_dir, split, "frost", classes, ignore_classes, idx_map)
        return dataset



@dataset_register(
    name='VQAv2_split1_c_fog', 
    classes=all_classes[0: 100], 
    task_type='Visual Question Answering',
    object_type='Generic Object',
    class_aliases=[],
    shift_type=None
)
class VQAv2_split1_c_fog(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose], 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        
        if transform is None:
            transform = None
            self.transform = transform
        dataset = _VQAv2_split1_c(root_dir, split, "fog", classes, ignore_classes, idx_map)
        return dataset



@dataset_register(
    name='VQAv2_split1_c_brightness', 
    classes=all_classes[0: 100], 
    task_type='Visual Question Answering',
    object_type='Generic Object',
    class_aliases=[],
    shift_type=None
)
class VQAv2_split1_c_brightness(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose], 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        
        if transform is None:
            transform = None
            self.transform = transform
        dataset = _VQAv2_split1_c(root_dir, split, "brightness", classes, ignore_classes, idx_map)
        return dataset



@dataset_register(
    name='VQAv2_split1_c_contrast', 
    classes=all_classes[0: 100], 
    task_type='Visual Question Answering',
    object_type='Generic Object',
    class_aliases=[],
    shift_type=None
)
class VQAv2_split1_c_contrast(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose], 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        
        if transform is None:
            transform = None
            self.transform = transform
        dataset = _VQAv2_split1_c(root_dir, split, "contrast", classes, ignore_classes, idx_map)
        return dataset



@dataset_register(
    name='VQAv2_split1_c_elastic_transform', 
    classes=all_classes[0: 100], 
    task_type='Visual Question Answering',
    object_type='Generic Object',
    class_aliases=[],
    shift_type=None
)
class VQAv2_split1_c_elastic_transform(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose], 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        
        if transform is None:
            transform = None
            self.transform = transform
        dataset = _VQAv2_split1_c(root_dir, split, "elastic_transform", classes, ignore_classes, idx_map)
        return dataset



@dataset_register(
    name='VQAv2_split1_c_pixelate', 
    classes=all_classes[0: 100], 
    task_type='Visual Question Answering',
    object_type='Generic Object',
    class_aliases=[],
    shift_type=None
)
class VQAv2_split1_c_pixelate(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose], 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        
        if transform is None:
            transform = None
            self.transform = transform
        dataset = _VQAv2_split1_c(root_dir, split, "pixelate", classes, ignore_classes, idx_map)
        return dataset



@dataset_register(
    name='VQAv2_split1_c_jpeg_compression', 
    classes=all_classes[0: 100], 
    task_type='Visual Question Answering',
    object_type='Generic Object',
    class_aliases=[],
    shift_type=None
)
class VQAv2_split1_c_jpeg_compression(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose], 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        
        if transform is None:
            transform = None
            self.transform = transform
        dataset = _VQAv2_split1_c(root_dir, split, "jpeg_compression", classes, ignore_classes, idx_map)
        return dataset



@dataset_register(
    name='VQAv2_split1_c_speckle_noise', 
    classes=all_classes[0: 100], 
    task_type='Visual Question Answering',
    object_type='Generic Object',
    class_aliases=[],
    shift_type=None
)
class VQAv2_split1_c_speckle_noise(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose], 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        
        if transform is None:
            transform = None
            self.transform = transform
        dataset = _VQAv2_split1_c(root_dir, split, "speckle_noise", classes, ignore_classes, idx_map)
        return dataset



@dataset_register(
    name='VQAv2_split1_c_gaussian_blur', 
    classes=all_classes[0: 100], 
    task_type='Visual Question Answering',
    object_type='Generic Object',
    class_aliases=[],
    shift_type=None
)
class VQAv2_split1_c_gaussian_blur(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose], 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        
        if transform is None:
            transform = None
            self.transform = transform
        dataset = _VQAv2_split1_c(root_dir, split, "gaussian_blur", classes, ignore_classes, idx_map)
        return dataset



@dataset_register(
    name='VQAv2_split1_c_spatter', 
    classes=all_classes[0: 100], 
    task_type='Visual Question Answering',
    object_type='Generic Object',
    class_aliases=[],
    shift_type=None
)
class VQAv2_split1_c_spatter(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose], 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        
        if transform is None:
            transform = None
            self.transform = transform
        dataset = _VQAv2_split1_c(root_dir, split, "spatter", classes, ignore_classes, idx_map)
        return dataset



@dataset_register(
    name='VQAv2_split1_c_saturate', 
    classes=all_classes[0: 100], 
    task_type='Visual Question Answering',
    object_type='Generic Object',
    class_aliases=[],
    shift_type=None
)
class VQAv2_split1_c_saturate(ABDataset):    
    def create_dataset(self, root_dir: str, split: str, transform: Optional[Compose], 
                       classes: List[str], ignore_classes: List[str], idx_map: Optional[Dict[int, int]]):
        
        if transform is None:
            transform = None
            self.transform = transform
        dataset = _VQAv2_split1_c(root_dir, split, "saturate", classes, ignore_classes, idx_map)
        return dataset


