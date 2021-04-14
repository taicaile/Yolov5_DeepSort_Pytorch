import collections
import sys
import json
sys.path.insert(0, './yolov5')

from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device, time_synchronized
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn

from count import object_cross_line

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def bbox_rel(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def get_optimal_font_scale(minwidth):
    return max(minwidth/75, 0.75)

def draw_boxes(img, bbox, identities=None, offset=(0, 0), classes_names=None):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        cls_name = classes_names[i][:3] if classes_names else ''
        label = '{}{:d}'.format(cls_name, id)
        optimal_font_scale = get_optimal_font_scale(min(abs(x2-x1), abs(y2-y1)))
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, optimal_font_scale, 1)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 +
                                 t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, optimal_font_scale, [255, 255, 255], 1)
    return img

def draw_count(img, count:dict, x1,y1):
    h,w,_= img.shape # h,w,c

    fontscale=min(3, h//300)
    thickness=2

    for name,num in count.items():
        label = f"{str(name):>10s}:{num}"
        label_width, label_height = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, fontscale, thickness)[0]
        cv2.putText(img, label, (x1, y1+label_height//2), cv2.FONT_HERSHEY_PLAIN, fontscale, [0,0,255], thickness)
        y1+=int(label_height*1.5)
    return img

def draw_track_history(img, xys, color=[0,0,255]):
    h,w,c = img.shape
    radius = min(h,w)//200
    for x,y in xys:
        cv2.circle(img,(x,y),color=color, radius=radius, thickness=-1)
        # img[y-p_size:y+p_size+1,x-p_size:x+p_size+1]=color

class TrackHistory:
    def __init__(self, max_size=20) -> None:
        self.max_size = max_size
        self.track_points = collections.defaultdict(list)
        self.tracked_ids_frame = set()
    def reset(self):
        self.track_points.clear()
        self.tracked_ids_frame.clear()

    def _get_key(self, cls,track_id):
        return f"{cls}:{track_id}"

    def add_point(self, cls, track_id, xy):
        key = self._get_key(cls, track_id)
        self.track_points[key].append(xy)
        if len(self.track_points[key])>self.max_size:
            self.track_points[key].pop(0)
        self.tracked_ids_frame.add(key)
        
    def pop(self,key):
        if self.track_points[key].__len__()>0:
            self.track_points[key].pop(0)
        else:
            self.track_points.pop(key)

    def check_miss_track(self):
        if self.tracked_ids_frame:
            track_miss = set(self.track_points.keys())-self.tracked_ids_frame
            for key in track_miss:
                self.pop(key)
        else:
            for key in self.track_points.keys():
                self.pop(key)

    def draw_all_points(self, img):

        for key,values in self.track_points.items():
            id = int(key.split(':')[1])
            color = compute_color_for_labels(id)
            draw_track_history(img, values, color)

class Line:
    def __init__(self,a,b,name='') -> None:
        self.a = a
        self.b = b
        self.name = name

class CrossLine:
    def __init__(self, names) -> None:
        self.names = names
        self.tracks_cur = set()
        self.tracks = {}
        self.count = {}
        self.missed = {}
        self.lines = None
        # missed over 5 time, delete track history
        self.miss_thr = 5 
    def load_json(self, file):
        with open(file, 'r') as f:
            j = json.load(f)
            lines = j['lines']
            names = set([line['name'] for line in lines])
            if len(names)!=len(lines):
                print("names are not set for all line, reset names...")
                name = 'A'
                for line in lines:
                    line['name']=name
                    name = chr(ord(name)+1)
            return lines
    def init(self, cfg):
        if not os.path.exists(cfg):
            self.running = False
            return
        self.running = True
        self.tracks.clear()
        self.count.clear()
        self.missed.clear()
        self.tracks_cur.clear()
        self.lines = self.load_json(cfg)
        for line in self.lines:
            self.tracks[line['name']] = collections.defaultdict(list)
            self.count[line['name']] = collections.defaultdict(int)
            self.missed[line['name']] = collections.defaultdict(int)

    def _get_key(self, cls, track_id):
        return f"{cls}:{track_id}"

    def add_track(self, cls, track_id, xyxy):
        if not self.running:
            return
        key = self._get_key(cls, track_id)
        self.tracks_cur.add(key)
        for line in self.lines:
            is_cross = object_cross_line(line['a'],line['b'],xyxy)
            cur_track = self.tracks[line['name']][key]
            cur_track.append(is_cross)
            if len(cur_track)>=2 and cur_track[-1]==True and not any(cur_track[:-1]):
                self.count[line['name']][self.names[cls]]+=1

    def check_miss_tracks(self):
        if not self.running:
            return
        for line in self.lines:
            tracks = self.tracks[line['name']]
            missed_tracks = set(tracks.keys()) - self.tracks_cur
            for key in missed_tracks:
                self.missed[line['name']][key]+=1
                if self.missed[line['name']][key]>self.miss_thr:
                    self.missed[line['name']].pop(key)


    def plot(self, img):
        if not self.running:
            return
        h,w,_= img.shape # h,w,c
        for line in self.lines:
            # plot line
            cv2.circle(img, tuple(line['a']), 5, (0, 0, 255), -1) 
            cv2.circle(img, tuple(line['b']), 5, (0, 0, 255), -1) 
            cv2.line(img=img, pt1=tuple(line['a']), pt2=tuple(line['b']),
                     color=(255, 0, 0), thickness=2)
            # plot count total near point a
            fontscale=min(3, h//160)
            thickness=2
            label = str(sum(self.count[line['name']].values()))
            label_width, label_height = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 
                                                        fontscale, thickness)[0]
            x,y = line['a']
            x = max(label_width//2, x)
            x = min(w-label_width, x)
            cv2.putText(img, label, (x,y-label_height//2), cv2.FONT_HERSHEY_PLAIN, fontscale, [0,0,255], thickness)
            x1,y1 = min(line['a'][0], line['b'][0]), h//20
            draw_count(img, self.count[line['name']],x1,y1)

def detect(opt, save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = torch.load(weights, map_location=device)[
        'model'].float()  # load to FP32
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        # view_img = True
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    # run once
    _ = model(img.half() if half else img) if device.type != 'cpu' else None

    save_path = str(Path(out))
    txt_path = str(Path(out)) + '/results.txt'

    seen_cls = {}
    trackhis = TrackHistory()
    crossline = CrossLine(names)
    preframe = None
    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        if vid_cap:
            curframe = vid_cap.get(cv2.CAP_PROP_POS_FRAMES)
            # detect new video, then reset deepsort trackers
            if not preframe:
                # first video
                crossline.init(Path(path).with_suffix(".json"))
            if preframe and curframe<preframe:
                print("new video detected, reset deepsort!!! ")
                deepsort.reset()
                trackhis.reset()
                crossline.init(Path(path).with_suffix(".json"))
                seen_cls = {}
            preframe = curframe

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]  # print string
            save_path = str(Path(out) / Path(p).name)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                bbox_xywh = []
                confs = []
                classes = []
                # Adapt detections to deep sort input format
                for *xyxy, conf, cls in det:
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(*xyxy)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    bbox_xywh.append(obj)
                    confs.append([conf.item()])
                    classes.append(int(cls.item()))

                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)

                # Pass detections to deepsort
                # [x1, y1, x2, y2, track_id]
                outputs = deepsort.update(xywhs, confss, classes, im0)
                trackhis.tracked_ids_frame.clear()
                crossline.tracks_cur.clear()
                
                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    # track id
                    identities = outputs[:, -2]
                    classes = outputs[:, -1]
                    classes_names = [names[cls] for cls in classes] if names else None
                    for i,cls in enumerate(classes):
                        track_id = identities[i] # 这里的track_id不是针对所有类别的总id，而是每个类别的id
                        seen_cls[names[cls]] = max(track_id, seen_cls.get(names[cls],0))
                        
                        # track history
                        x1,y1,x2,y2 = bbox_xyxy[i]
                        trackhis.add_point(cls,track_id,[(x1+x2)//2, (y1+y2)//2])

                        # cross detection line
                        crossline.add_track(cls,track_id,bbox_xyxy[i])
                        

                    draw_boxes(im0, bbox_xyxy, identities, classes_names=classes_names)
                    # draw_count(im0, seen_cls)
            
                trackhis.check_miss_track()
                trackhis.draw_all_points(im0)
                crossline.check_miss_tracks()
                crossline.plot(im0)
                # Write MOT compliant results to file
                if save_txt and len(outputs) != 0:
                    for j, output in enumerate(outputs):
                        bbox_left = output[0]
                        bbox_top = output[1]
                        bbox_w = output[2]
                        bbox_h = output[3]
                        identity = output[-1]
                        with open(txt_path, 'a') as f:
                            f.write(('%g ' * 10 + '\n') % (frame_idx, identity, bbox_left,
                                                           bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))  # label format

            else:
                deepsort.increment_ages()

            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                # cv2.imshow(p, im0)
                cv2.imshow('im0', im0)

                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                print('saving img!')
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    print('saving video!')
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        default='yolov5/weights/yolov5s.pt', help='model.pt path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str,
                        default='inference/images', help='source')
    parser.add_argument('--output', type=str, default='inference/output',
                        help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v',
                        help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    # class 0 is person
    parser.add_argument('--classes', nargs='+', type=int,
                        default=[0], help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument("--config_deepsort", type=str,
                        default="deep_sort_pytorch/configs/deep_sort.yaml")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)
    print(args)

    with torch.no_grad():
        detect(args)
