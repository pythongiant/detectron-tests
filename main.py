import detectron2

print(f"Detectron2 version is {detectron2.__version__}")

from detectron2.engine import DefaultPredictor
from detectron2.config import  get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2
import requests
import numpy as np  

# load an image of Lionel Messi with a ball
image_reponse = requests.get("https://lh3.googleusercontent.com/wuZkUL2JstwXl8b9nNe16G0TBO9K-eypsF_b9M8VxPF__1JX1qtTPvm-P4xKkfmbp5vjVQRlkpi4TlkkB8-xT9-9YksaPPMBam_RhScasS74HolQQPeq1mPJm50lnFnQT2_ZrnPkw0Jv7K9M3PzUpKdvyBniXcPxZxQUom0BfIf29aLsOnY86T6BeMVGj75XmCxRLej8ZFTzNYWZkj3M8_0u6aGi0zDRkvs7gffWTpfUrT1IF8KDwQJCq-IeOxvONH-j2DjkOBd0z4ALfyaUlONwIluIY0jRv6wMlVu4ZwnpVL5BCgfZSXoeFC8c_0zJhZDhGg4tWpoACYkQrLMc6IhGesxry7-ymQuJ70VhMFD62X3Ct2arfOPVDzc8Pn3Z7r7vlZMmRm1UOv_ujqYhewx1PnKdRUp4qJd9_1gBDcIVyZcgiRgUSa1a_FhiBQSnCbDCpOJy-RodGs_RGtkirIqL8zEWuWgKFRHQSCedyX7AKsrHPTMgDa28MQqw7WGYKAOWRxlEqfr4bkTjv9AJ9FGHP3feNxiHssKBIHhQIAG9mQ1wzSdiqEOy9IkvtjAhJV0Uk-4GvvkEWJXpAoXwrf-c8VP82VXhM5d85PQ9WshPWgpN3zCi7Yl3aM3kI5ibu-ZkgQJ33o9-1HRlkL39ndOp_8AZBc-hBZTX1f6DKDmbSgm3t22AJ_e2s0NrTQJ3XeQqH6Z453Znq5RoiGFCtt29IWZZ8tQOdWyaghNbVBrd78jNgydsmKp_I1bblh5vLzwDPOjS04vtcJJbufer6AR71jgbfZ-C=w670-h893-no?authuser=0")
image_as_np_array = np.frombuffer(image_reponse.content, np.uint8)
image = cv2.imdecode(image_as_np_array, cv2.IMREAD_COLOR)


cfg = get_cfg()
cfgFile = "./config.yaml"
cfg.merge_from_file(cfgFile)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x/137260431/model_final_a54504.pkl"
cfg.MODEL.DEVICE = "cpu"


predictor = DefaultPredictor(cfg)

# make prediction

output = predictor(image)
print(output) 

v = Visualizer(image[:,:,::-1],MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),scale=1.2)
out = v.draw_instance_predictions(output["instances"].to("cpu"))
cv2.imwrite("out.jpg",out.get_image()[:,:,::-1])
