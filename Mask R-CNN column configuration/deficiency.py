import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
from matplotlib import pyplot as plt
from mrcnn.config import Config
from mrcnn import model as modellib, utils
import math

# Root directory of the project
ROOT_DIR = os.path.abspath("./")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
print(COCO_WEIGHTS_PATH)

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class DeficiencyConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "deficiency"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + face

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 300

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.8

############################################################
#  Dataset
############################################################

class DeficiencyDataset(utils.Dataset):
    
    def load_deficiency(self, dataset_dir, subset):
        """Load a subset of the Deficiency dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have 4 class to add.
        self.add_class("deficiency", 1, "crack")
        #self.add_class("spalling", 2, "spalling")
        #self.add_class("horizontal", 3, "horizontal")
        #self.add_class("vertical", 4, "vertical")
        
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        
        # Load annotations
        # LabelBox
        # { "ID": "...",
        #     "DataRow ID": "...",
        #     "Labeled Data": "...",
        #     "Label":{
        #         "objects": [
        #             {
        #                 "featureId": "ckfolp6hb0iu10y6heyyzgzpp",
        #                 "schemaId": "ckfn12xqv058l0yb2hsm0akbj",
        #                 "title": "crack",
        #                 "value": "crack",
        #                 "color": "#1CE6FF",
        #                 "line":[
        #                     {
        #                         "x": 289.379,
        #                         "y": 320.00
        #                     } 
        #                    ...
        #                 ],
        #                 "instanceURI": "..."
        #             },
        #             .. more regions ..
        #         ],
        #         "classifications": []
        #     },
        #     "Created By": "youjeong.jang@jacks.sdstate.edu",
        #     "Project Name": "deficiency",
        #     "Created At": "2020-09-28T21:26:17.000Z",
        #     "Updated At": "2020-10-28T00:52:52.000Z",
        #     "Seconds to Label": 4114.295,
        #     "External ID": "CIP Test 039.jpg",
        #     "Agreement": -1,
        #     "Benchmark Agreement": -1,
        #     "Benchmark ID": null,
        #     "Dataset Name": "CIP_TEST_Pics",
        #     "Reviews": [],
        #     "View Label": "https://editor.labelbox.com?project=ckfiu6og51qlr07096p58xi39&label=ckfn1kjte00003b60pj9awy53"
        #     }
        annotations = json.load(open(os.path.join("C:/Users/kwon/DeepSegmentor","label.json")))

        # Label box is already list, dict type need this one
        # annotations = list(annotations.values())
        for a in annotations:
            label = a['Label']['objects']
            print(a['External ID'])

            poly = []
            label_ids = []
            label_dict = {"crack": 1, "spalling": 2, "horizontal": 3, "vertical": 4}
            for r in label:
                if r['title'] == 'crack' and r['line']: # class is crack and also this should have line attribute if not false
                    pair = [[i['x'], i['y']] for i in r['line']]
                    # x = [i['x'] for i in r['line']]
                    # y = [i['y'] for i in r['line']]
                    # print(pair)
                    poly.append({"name":"line", "pair":pair})
                    label_ids.append(label_dict["crack"])
            image_path = os.path.join(dataset_dir, a['External ID'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]
            self.add_image(
                "deficiency",
                image_id = a['External ID'],
                path = image_path,
                width=width, height=height,
                polygons = poly,
                num_ids = label_ids)
            
    def load_deficiency_crop(self, dataset_dir, subset):
        print("load deficiency crop in")
        # Add classes. We have 4 class to add.
        self.add_class("deficiency", 1, "crack")
        #self.add_class("spalling", 2, "spalling")
        #self.add_class("horizontal", 3, "horizontal")
        #self.add_class("vertical", 4, "vertical")
        
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        
        annotations = json.load(open(os.path.join("C:/Users/kwon/DeepSegmentor","label.json")))
        
        n_img_size_w = 544
        n_img_size_h = 384
        for a in annotations: # each images
            label = a['Label']['objects']
            print(a['External ID'])
            
            image_path = os.path.join(dataset_dir,a['External ID'])
            image = skimage.io.imread(image_path)

            oh, ow = image.shape[:2]
            i, j = math.floor(image.shape[0]/n_img_size_h), math.floor(image.shape[1]/n_img_size_w)
            print("Original h,w: ", oh, ow, "Quot: ", i, j)
    
            ci = n_img_size_h
            cj = n_img_size_w
    
            for m in range(i):
                for n in range(j):
                    # print((n_img_size*m) ,(n_img_size*(m+1)))
                    new_h1, new_h2 = (n_img_size_h*m), (n_img_size_h*(m+1))
                    new_w1, new_w2 = (n_img_size_w*n), (n_img_size_w*(n+1))
                    new_image = image[new_h1 : new_h2, new_w1 : new_w2]
            
                    #new_image save
                    #new_image name save
                    new_filename = a['External ID'].split('.')[0] + "_" + str(new_h1) + str(new_w1) + ".png"
                    print(dataset_dir+"/"+new_filename)
                    skimage.io.imsave(dataset_dir+"/"+new_filename, new_image)
                    # skimage.io.imsave("C:/Users/kwon/DeepSegmentor/datasets/sample/val/"+new_filename, new_image)
            
                    poly = []
                    label_ids = []
                    label_dict = {"crack": 1, "spalling": 2, "horizontal": 3, "vertical": 4}
                    for r in label:
                        pair = []
                        if r['title'] == 'crack' and r['line']: # class is crack and also this should have line attribute if not false
                            for i in r['line']:
                                if ((i['y'] >= new_h1 and i['y'] <= new_h2) and (i['x'] >= new_w1 and i['x'] <= new_w2)):
                                    # print([i['x']-new_w1, i['y']-new_h1])
                                    pair.append([i['x']-new_w1, i['y']-new_h1])

                            # print("pair: ", pair)
                            if len(pair) > 0: 
                                poly.append({"name":"line", "pair":pair})
                                label_ids.append(label_dict["crack"])
                            # print("poly: ", poly)
                    print(label_ids)
                    image_path = os.path.join(dataset_dir, new_filename)
                    print(new_filename)
            
                    self.add_image(
                        "deficiency",
                        image_id = new_filename,
                        path = image_path,
                        width=n_img_size_w, height=n_img_size_h,
                        polygons = poly,
                        num_ids = label_ids)
        
            
    def load_mask(self, image_id):
        """
        Generate instance masks for an image.
        Returns:
            labeled_image: A bool array of shape [height, width, instance count] with one mask per instance.
            label_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        if image_info["source"] != "deficiency":
            return super(self.__class__, self).load_mask(image_id)
        
        info = self.image_info[image_id]
        if info["source"] != "deficiency":
            return super(self.__class__, self).load_mask(image_id)
        
        # print("info: ", info)
        label_ids = info['num_ids']
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],dtype=np.int32)
        
        for i, p in enumerate(info["polygons"]):
            tmp = np.zeros([info["height"], info["width"]],dtype=np.int32)
            npa = np.asarray(p['pair'], dtype=np.int32)
            mask[:,:,i] = cv2.polylines(tmp, [npa], False, (255, 255, 255)) / 255
        
        label_ids = np.array(label_ids, dtype=np.int32)
        print(mask.shape)
        return mask.astype(np.bool), label_ids.astype(np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "deficiency":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
            
def train(model):
    """Train the model."""
    # Training dataset.
    print("deficiency training start: ")
    dataset_train = DeficiencyDataset()
    dataset_train.load_deficiency_crop(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = DeficiencyDataset()
    dataset_val.load_deficiency_crop(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=50,
                layers='heads')


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)

############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect balloons.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/face/dataset/",
                        help='Directory of the face dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = DeficiencyConfig()
    else:
        class InferenceConfig(DeficiencyConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
