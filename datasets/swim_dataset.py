import torch.utils.data as data
from utils.augmentlandmark import *
from utils.bbox import constraint_theta


class SWIMDataset(data.Dataset):

    def __init__(self,
                 dataset=None,
                 augment=False,
                 ):
        self.image_set_path = dataset
        if dataset is not None:
            self.image_list = self._load_image_names()
        self.classes = ('__background__', 'wake')
        self.num_classes = len(self.classes)
        self.class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self.augment = augment

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        im_path = self.image_list[index]
        im = cv2.cvtColor(cv2.imread(im_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        roidb = self._load_annotation(self.image_list[index])
        ldmdb = self._load_landmark(self.image_list[index])
        gt_inds = np.where(roidb['gt_classes'] != 0)[0]
        nt = len(roidb['boxes'])
        gt_boxes = np.zeros((len(gt_inds), 6), dtype=np.float32)
        gt_landmarks = np.zeros((len(gt_inds), 4), dtype=np.float32)
        if nt:
            bboxes = roidb['boxes'][gt_inds, :]
            classes = roidb['gt_classes'][gt_inds]
            landmarks = ldmdb['landmarks'][gt_inds, :]
            if self.augment:
                transform = Augment([HSV(0.5, 0.5, p=0.5),
                                     HorizontalFlip(0.5),
                                     VerticalFlip(0.5),
                                     Affine(degree=10, translate=0.1, scale=0.1, p=0.5)],
                                    box_mode='xywha', )
                im, bboxes, landmarks = transform(im, bboxes, landmarks)
            gt_boxes[:, :-1] = bboxes
            gt_landmarks = landmarks

            mask = mask_valid_boxes(bboxes, return_mask=True)
            bboxes = bboxes[mask]
            gt_boxes = gt_boxes[mask]
            classes = classes[mask]
            gt_landmarks = gt_landmarks[mask]

            for i, bbox in enumerate(bboxes):
                gt_boxes[:, 5] = classes[i]
            gt_boxes = constraint_theta(gt_boxes)
            cx, cy, w, h = [gt_boxes[:, x] for x in range(4)]
            x1 = cx - 0.5 * w
            x2 = cx + 0.5 * w
            y1 = cy - 0.5 * h
            y2 = cy + 0.5 * h
            gt_boxes[:, 0] = x1
            gt_boxes[:, 1] = y1
            gt_boxes[:, 2] = x2
            gt_boxes[:, 3] = y2

        return {'image': im, 'boxes': gt_boxes, 'landmarks': gt_landmarks, 'path': im_path}

    def _load_image_names(self):
        image_set_file = self.image_set_path
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_list = [x.strip() for x in f.readlines()]
        return image_list

    def _load_annotation(self, index):
        root_dir, img_name = os.path.split(index)
        filename = os.path.join(root_dir.replace('JPEGImages', 'Annotations'), img_name[:-4] + '.xml')
        boxes, gt_classes = [], []
        with open(filename, 'r', encoding='utf-8-sig') as f:
            content = f.read()
            assert '<object>' in content, 'Background picture occurred in %s' % filename
            objects = content.split('<object>')
            info = objects.pop(0)
            for obj in objects:
                assert len(obj) != 0, 'No onject found in %s' % filename
                diffculty = obj[obj.find('<difficult>') + 11: obj.find('</difficult>')]
                if diffculty == '1':
                    continue
                cx = round(eval(obj[obj.find('<cx>') + 4: obj.find('</cx>')]))
                cy = round(eval(obj[obj.find('<cy>') + 4: obj.find('</cy>')]))
                w = round(eval(obj[obj.find('<w>') + 3: obj.find('</w>')]))
                h = round(eval(obj[obj.find('<h>') + 3: obj.find('</h>')]))
                a = eval(obj[obj.find('<angle>') + 7: obj.find('</angle>')]) / math.pi * 180
                box = np.array([cx, cy, w, h, a])

                box[4] = box[4] * (box[4] <= 90) + (box[4] - 180) * (box[4] > 90)
                boxes.append(box)
                label = 1
                gt_classes.append(label)
        return {'boxes': np.array(boxes), 'gt_classes': np.array(gt_classes)}

    def _load_landmark(self, index):
        root_dir, img_name = os.path.split(index)
        filename = os.path.join(root_dir.replace('JPEGImages', 'Landmarks'), img_name[:-4] + '.xml')
        landmarks = []
        with open(filename, 'r', encoding='utf-8-sig') as f:
            content = f.read()
            assert '<object>' in content, 'Background picture occurred in %s' % filename
            objects = content.split('<object>')
            info = objects.pop(0)
            for obj in objects:
                assert len(obj) != 0, 'No onject found in %s' % filename
                diffculty = obj[obj.find('<difficult>') + 11: obj.find('</difficult>')]
                if diffculty == '1':
                    continue
                px = round(eval(obj[obj.find('<px>') + 4: obj.find('</px>')]))
                py = round(eval(obj[obj.find('<py>') + 4: obj.find('</py>')]))
                theta1 = eval(obj[obj.find('<theta1>') + 8: obj.find('</theta1>')]) / math.pi * 180
                theta2 = eval(obj[obj.find('<theta2>') + 8: obj.find('</theta2>')]) / math.pi * 180
                landmark = np.array([px, py, theta1, theta2])
                landmarks.append(landmark)
        return {'landmarks': np.array(landmarks)}

    def return_class(self, id):
        id = int(id)
        return self.classes[id]
