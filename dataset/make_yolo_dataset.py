import os
import json
import shutil

DATAROOT = "./"
ALLJSON = os.path.join(DATAROOT,"train_pseudo.json")
TRAINJSON = os.path.join(DATAROOT,"train_pseudo_k4.json")
VALIDJSON = os.path.join(DATAROOT,"valid_4.json")
IMAGEPATH = os.path.join(DATAROOT,"train/")

'''
Convert COCO Dataset to Yolo Dataset
box = [x, y, w, h]
return = x_centor, y_centor, w, h(normalized)
'''
def _convert(box):
	dw = 1./1024
	dh = 1./1024
	x, y, w, h = box
	x_c = (x + x + w) / 2.0
	y_c = (y + y + h) / 2.0
	x = x_c*dw
	w = w*dw
	y = y_c*dh
	h = h*dh
	return (x, y, w, h)

'''
Make yolo dataset
'''
def _make_yolo_dataset(images,json_dir, image_dir, label_dir):
	with open(json_dir, "r", encoding="utf8") as outfile:
		json_data = json.load(outfile)
	yolo_images = json_data["images"]

	for yolo_image in yolo_images:
		id = yolo_image['id']
		img = images[id]
		string = ''
		for bbox, cate in zip(img['bbox'], img['bbox_category']):
			x, y, w, h = _convert(bbox)
			string += f"{cate} {x} {y} {w} {h}\n"

		with open(os.path.join(label_dir,f"{img['id']:04}.txt"), 'w') as f:
			f.write(string.strip())

		shutil.copyfile(os.path.join(img['file_name']), os.path.join(image_dir,f"{img['id']:04}.jpg"))


'''
Make dir
'''
def _make_directory(paths):
	for path in paths:
		os.makedirs(path, exist_ok=True)


'''
Wrap func
'''
def make(images,json,path):
	imagePath = DATAROOT+'yolov5_pseudo_k4/images/'+path
	labelPath = DATAROOT+'yolov5_pseudo_k4/labels/'+path

	_make_directory([imagePath,labelPath])
	_make_yolo_dataset(images,json,imagePath,labelPath)


'''
Main
'''
def __main__():
	with open(ALLJSON, "r", encoding="utf8") as outfile:
			json_data = json.load(outfile)

	images = json_data["images"]
	annotations = json_data["annotations"]
	for image in images:
		image['bbox'] = []
		image['bbox_category'] = []

	for annotation in annotations:
		image_id = annotation["image_id"]
		category_id = annotation["category_id"]
		bbox = annotation["bbox"]

		images[image_id]['bbox'].append(bbox)
		images[image_id]['bbox_category'].append(category_id)
	

	make(images,TRAINJSON, 'train')
	make(images,VALIDJSON, 'valid')


if __name__=='__main__':
	__main__()