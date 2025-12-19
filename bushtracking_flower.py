from ultralytics import YOLO # type: ignore
import pandas as pd # type: ignore
import numpy as np
import argparse
import pysrt # type: ignore
import csv
import cv2
import os
import re

'''
This script takes in a directory name with video files (mp4) and 
their associated subtitle files (srt), a bush and berry model 
files (pt); and counts bushes from video as well as gathering 
other data like gps location, number of berries, and images of bushes
'''

'''
Here's file structure to run script:

> current directory
| > weights
| | bush.pt
| | berry.pt
| > videos
| | > dataset1
| | | video1.mp4
| | | video1.srt
| | | video2.mp4
| | | video2.srt
| | | etc...
| bushtracking.py

'''


# bush class for handling detections over multiple frames
class Bush: 
    def __init__(self, box:list[float]):
        self.frames = [box]
        self.lastupdated = 0 # number of frames since last on screen
        self.updated = True # whether bush is on current frame
        self.counted = False # whether bush was already counted or not
        self.ldscore = 0 # number of frames on screen - number of frames gone
        
    def add_frame(self, box:list[float]): # add bounding box for current frame
        self.frames.append(box)
        self.lastupdated = 0
        self.updated = True
        self.ldscore += 1
        
    def age(self):
        self.lastupdated += 1
        self.ldscore -= 1
    
    def mid_y(self, box:list[float]) -> float:
        return (box[1]+box[3])/2
        
    def mid_x(self, box:list[float]) -> float:
        return (box[0]+box[2])/2
        
    def moving_vote(self, fnl) -> int: 
        # compare where bush is now to where bush was
        inl = self.frames[0]
        if len(self.frames) > 5:
            inl = self.frames[-5]
                
        if fnl[0] < inl[0] and fnl[2] < inl[2]:
            return 1 # right
            
        if inl[0] < fnl[0] and inl[2] < fnl[2]:
            return -1 # left
            
        return 0
        
    def crossed(self, vid_width:int, vid_height:int, vid_direction:str) -> bool:
        # tell if bush has met criteria to be counted
        if self.counted: 
            return False
        
        if len(self.frames) < 2:
            return False
            
        if self.ldscore < 0:
            return False
        
        # bush should be near middle of screen vertically
        y = self.mid_y(self.frames[-1]) / vid_height
        if y < 0.25 or y > 0.75:
            return False
        
        # check if bush crossed middle of screen horizontally
        x0 = self.mid_x(self.frames[-2]) / vid_width
        x1 = self.mid_x(self.frames[-1]) / vid_width    
        if vid_direction == "right":
            if x1 <= 0.5 and 0.5 <= x0:
                self.counted = True
                return True
        else:
            if x0 <= 0.5 and 0.5 <= x1:
                self.counted = True
                return True
            
        return False
        
    def count_flowers(self, img:cv2.Mat, flower_model:YOLO) -> tuple[int, cv2.Mat]:
        # img is segment of frame including only the bounding box of the bush
        height, width, channels = img.shape
        X, Y = [0], [0]
        flower_count = 0
        annotated_img = img
        annotated_columns = []
        
        while X[-1] < width:
            X.append(X[-1]+640)
             
        while Y[-1] < height:
            Y.append(Y[-1]+640)
        
        # do predictions in 648x648 tiles
        # exclude predictions with midpoints
        # outside x=640 or y=640
        # and then glue them all together again 
        for i in range(len(X)-1):
            annotated_tiles = []
            for j in range(len(Y)-1):
                tile = img[Y[j]:Y[j+1]+8,X[i]:X[i+1]+8]
                results = flower_model.predict(tile)
                try:
                    xyxy = results[0].boxes.xyxy.cpu().numpy().astype(int).tolist()
                except:
                    xyxy = []
                for box in xyxy:
                    ([left, top, rite, btm]) = box
                    if left + rite < 1280 and top + btm < 1280:
                        flower_count += 1
                        color = (148, 247, 175)
                        tile = cv2.rectangle(tile, (left, top), (rite, btm), color, 1)
                
                tile = tile[0:640, 0:640]
                annotated_tiles.append(tile)
                
            annotated_column = annotated_tiles[0]
            del annotated_tiles[0]
            for annotated_tile in annotated_tiles:
                annotated_column = cv2.vconcat([annotated_column,annotated_tile])
            
            annotated_columns.append(annotated_column)
        
        annotated_img = annotated_columns[0]
        del annotated_columns[0]
        for annotated_column in annotated_columns:
            annotated_img = cv2.hconcat([annotated_img, annotated_column])
        
        return flower_count, annotated_img

def parse_srt(subs):
    gps_data = []
    for sub in subs:
        time_stamp = (sub.start.hours * 3600 + 
                      sub.start.minutes * 60 + 
                      sub.start.seconds + 
                      sub.start.milliseconds / 1000.0)
        text = sub.text
        lat_match = re.search(r'latitude: ([\d\.\-]+)', text)
        lng_match = re.search(r'longitude: ([\d\.\-]+)', text)
        alt_match = re.search(r'rel_alt: ([\d\.\-]+)', text)
        if lat_match and lng_match and alt_match:
            lat = float(lat_match.group(1))
            lng = float(lng_match.group(1))
            alt = float(alt_match.group(1))
            data = (time_stamp, lat, lng, alt)
            gps_data.append(data)
            
    return gps_data
    
def get_video_names(source:str):
    files = os.listdir(f"videos/{source}")
    files = [file.split(".") for file in files]
    files = [file for file in files if len(file) > 1]
    video_names = [file[0] for file in files if file[1] == "mp4"]
    video_names.sort()
    return video_names

def get_row_num(name:str):
    row_match = re.search(r'(?i)row:?[_\s]*?([\d]+)', name)
    if not row_match:
        return None
    
    return int(row_match.group(1))

def get_row_numbers(names):
    rownumbers = [get_row_num(name) for name in names]
    mex = 1
    while None in rownumbers:
        while mex in rownumbers:
            mex += 1
        
        rownumbers[rownumbers.index(None)] = mex

    return rownumbers

    
    


def main(source:str, kwargs:dict):

    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    if not source:
        print('No path for video(s) provided')
        return 1
    
    if not os.path.exists(f'videos/{source}'):
        print('Path to video(s) not found')
        return 1

    names = get_video_names(source)
    rownumbers = get_row_numbers(names)

    weights = os.listdir('weights')
    models = {m:[w for w in weights if m in w][0] for m in ('bush', 'flower')}
    bush_model = YOLO(f'weights/{models["bush"]}')
    flower_model = YOLO(f'weights/{models["flower"]}')

    csv_header = ['filename', 
                  'frame', 
                  'id',
                  'bushnumber',
                  'rownumber',
                  'latitude', 
                  'longitude',
                  'flowercount']
    csv_data = [csv_header]

    for name, i in zip(names, rownumbers):
        csv_data += do_tracking(source, name, i, (bush_model, flower_model), kwargs)

    with open(f"./videos/{source}/{source}.csv", 'w', newline = '') as file:
        writer = csv.writer(file)
        writer.writerows(csv_data)


def do_tracking(source:str, name:str, row_number:int, models:tuple[YOLO, YOLO], kwargs:dict) -> list[list[str]]:
    FRAME_STOP = kwargs.get('frame_stop', None)
    PROCESS_FPS = kwargs.get('frame_rate', None)
    save_frames = kwargs.get('save_frames', False)
    save_video = kwargs.get('save_video', False)
    CONF = kwargs.get('conf', 0.5)
    IOU = kwargs.get('iou', 0.15)
    verbose = kwargs.get('verbose', False)
    bush_model, flower_model = models

    cap = cv2.VideoCapture(f"videos/{source}/{name}.mp4")
    subs = pysrt.open(f"videos/{source}/{name}.srt")
    gps_data = parse_srt(subs)

    WIDTH  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS)
    DOWNSCALE = 4
    if not PROCESS_FPS:
        PROCESS_FPS = round(FPS)

    FPPF = round(FPS) / PROCESS_FPS

    frame_count = 0
    bush_count = 0
    registered_ids = {}
    direction_votes = 0
    dead_ids = []

    save_dir = f"videos/{source}/results_{name}"
    os.makedirs(f"{save_dir}/bushes", exist_ok=True)
    os.makedirs(f"{save_dir}/bushes_annotated", exist_ok=True)
    csv_data:list[list[str]] = []
    if save_video:
        video = cv2.VideoWriter(f"{save_dir}/{name}_results.avi", 
                                cv2.VideoWriter_fourcc(*'DIVX'), 
                                PROCESS_FPS, 
                                (WIDTH // DOWNSCALE, HEIGHT // DOWNSCALE))
        
    # figure out drone direction
    while cap.isOpened():
        # wait until we've colleted enough votes on which way drone is flying    
        success, frame = cap.read()
        if not success or (frame_count > 4 * FPS and 
                        abs(direction_votes) - len(registered_ids) > frame_count / 2):
            break
        # get detections for current frame    
        results = bush_model.track(frame,
                                conf=CONF,
                                iou=IOU,
                                persist=True)
        try:
            ids = results[0].boxes.id.cpu().numpy().astype(int).tolist()
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int).tolist()
            ids_boxes = dict(zip(ids,boxes))
            # add new ids, but dont bother updating old ones
            # just compare
            for id in ids:
                if id in registered_ids:
                    direction_votes += registered_ids[id].moving_vote(ids_boxes[id])
                else:
                    registered_ids[id] = Bush(ids_boxes[id])
        
        except Exception as e:
            print("Oops!")
            print(e)

        frame_count += 1

    direction = "left"
    if direction_votes > 0:
        direction = "right"
    print(f"Drone flying {direction}, " 
            f"with score {direction_votes} " 
            f"after {frame_count} frames.")
    
    # reset video to beginning
    registered_ids.clear()
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_count = 0

    # now count bushes
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # process video at specified frame rate    
        if frame_count % FPPF == 0:
            # reset for new frame
            dead_ids = [] 
            for bush in registered_ids.values():
                bush.updated = False
            
            # get bush detections for current frame
            results = bush_model.track(frame,
                                    conf=CONF,
                                    iou=IOU,
                                    persist=True)
            if save_video:
                video_frame = results[0].plot(color_mode="instance")
                video_frame = cv2.putText(video_frame,
                                          f"Frame {frame_count}",
                                          (100, HEIGHT  - 100),
                                          cv2.FONT_HERSHEY_SIMPLEX,
                                          4,
                                          (32, 0, 0),
                                          8)
                video_frame = cv2.resize(video_frame, 
                                         (WIDTH // DOWNSCALE, HEIGHT // DOWNSCALE))
                video.write(video_frame)
            
            try:
                ids = results[0].boxes.id.cpu().numpy().astype(int).tolist()
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int).tolist()
                ids_boxes = dict(zip(ids,boxes))
                # new id means new bush
                # update current ids, add new ids
                for id in ids:
                    if id in registered_ids:
                        registered_ids[id].add_frame(ids_boxes[id])
                    else:
                        registered_ids[id] = Bush(ids_boxes[id])
                
            except Exception as e:
                print("Oops!")
                print(e)
            
            # check all ids in registry    
            for id in registered_ids:
                bush:Bush = registered_ids[id]
                # if it was updated, it couldve crossed so check
                if bush.updated:
                    if bush.crossed(WIDTH,HEIGHT,direction):
                        # add bush to tally, get flower count, 
                        # collect csv data, save frame with bush and berry annotations
                        bush_count += 1
                        (left, top, rite, btm) = bush.frames[-1]
                        # thnx luke
                        img = frame.copy()
                        flower_count, img[top:btm, left:rite] = bush.count_flowers(img[top:btm, left:rite], flower_model)
                        # thnx jake
                        time_stamp = frame_count / FPS
                        closest_gps = min(gps_data, key=lambda x: abs(x[0] - time_stamp))
                        (time_stamp, lat, lng, alt) = closest_gps
                        img_name = f"frame{frame_count:04d}_id{id:04d}_lat{lat:.6f}_lng{lng:.6f}"
                        csv_data.append( [img_name, frame_count, id, bush_count, row_number, lat, lng, flower_count] )
                        if save_frames:
                            img = cv2.rectangle(img, (left,top), (rite,btm), (255, 0, 0), 12)
                            cv2.imwrite(f"{save_dir}/bushes_annotated/{img_name}.jpg", img)
                            cv2.imwrite(f"{save_dir}/bushes/{img_name}.jpg", frame)
                else:
                    # ids decay over time
                    bush.age()
                    # if it has been enough frames since last update, remove id from registry
                    if bush.lastupdated >= PROCESS_FPS / 2:
                        dead_ids.append(id)
                    
            for id in dead_ids:
                del registered_ids[id]
            
            # display info
            if verbose:
                print(f"Frame: {frame_count}")
                print(f"{len(registered_ids)} ids, {bush_count} counted.")
                for id in registered_ids:
                    bush = registered_ids[id]
                    print(f"{id:04d}:{str(bush.updated):<5},"
                        f"{bush.lastupdated:02d},"
                        f"{str(bush.counted):<5},"
                        f"{bush.frames[-1 % len(bush.frames)]}")
                        
            else:
                pass
        
        # stop early if needed    
        if FRAME_STOP and not frame_count < FRAME_STOP:
            break        
        
        frame_count += 1


    cap.release()
    if save_video:
        video.release()
        
    cv2.destroyAllWindows()
    return csv_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('source',
                        help=("Name of folder with " 
                        "mp4s and srts."))
    parser.add_argument('--frame_stop',
                        type=int,
                        help=("Optional. Frame number "
                        "to stop on."))
    parser.add_argument('--frame_rate',
                        type=int,
                        default=None,
                        help=("Optional. Sets frame rate "
                        "to process video. Must be a factor "
                        "of frame rate of video."))
    parser.add_argument('--save_frames', 
                        action="store_true",
                        help=("Optional. Saves frames of "
                        "bushes when enabled."))
    parser.add_argument('--save_video', 
                        action="store_true",
                        help="Optional. Creates video file.")
    parser.add_argument('--conf',
                        type=float,
                        default=0.50,
                        help=("Optional. Set the confidence "
                        "level for bush predictions. Default 0.50."))
    parser.add_argument('--iou',
                        type=float,
                        default=0.15,
                        help=("Optional. Set the max IOU "
                        "(overlap) for bush predictions. Default 0.15."))
    parser.add_argument('--verbose', 
                        action="store_true",
                        help="Optional. Gives frame by frame "
                        "summary when enabled.")
    
    args = parser.parse_args()
    kwargs = vars(args)
    source = args.source.replace("'",'')
    main(source, kwargs)