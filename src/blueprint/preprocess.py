import numpy as np
import cv2

def extract_frames(filename, num_frames, model, image_size=(380, 380)):
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        return [], []
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idxs = np.linspace(0, frame_count - 1, min(num_frames, frame_count), dtype=int)
    
    croppedfaces, idx_list = [], []
    for idx in frame_idxs:
        ret = cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        if not ret:
            continue
        ret, frame = cap.read()
        if not ret:
            continue
            
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = extract_face(frame, model, image_size)
        if faces:
            croppedfaces.extend(faces)
            idx_list.extend([idx] * len(faces))
            
    cap.release()
    return croppedfaces, idx_list

def extract_face(frame, model, image_size=(380, 380)):
    faces = model.predict_jsons(frame)
    if len(faces[0]['bbox']) == 0:
        scale_factor = 0.5
        small_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor)
        faces = model.predict_jsons(small_frame)
        if len(faces[0]['bbox']) == 0:
            return []
        for face in faces:
            face['bbox'] = [coord/scale_factor for coord in face['bbox']]
    
    croppedfaces = []
    for face_idx in range(len(faces)):
        x0, y0, x1, y1 = faces[face_idx]['bbox']
        bbox = np.array([[x0, y0], [x1, y1]])
        face = crop_face(frame, None, bbox, False, crop_by_bbox=True, only_img=True, phase='test')
        croppedfaces.append(cv2.resize(face, dsize=image_size).transpose((2, 0, 1)))
    
    return croppedfaces

def crop_face(img, landmark=None, bbox=None, margin=False, crop_by_bbox=True, abs_coord=False, only_img=False, phase='test'):
    assert phase in ['train', 'val', 'test']
    assert landmark is not None or bbox is not None
    
    H, W = len(img), len(img[0])
    
    if crop_by_bbox:
        x0, y0 = bbox[0]
        x1, y1 = bbox[1]
        w = x1 - x0
        h = y1 - y0
        w0_margin = w/4
        w1_margin = w/4
        h0_margin = h/4
        h1_margin = h/4
    else:
        x0, y0 = landmark[:68,0].min(), landmark[:68,1].min()
        x1, y1 = landmark[:68,0].max(), landmark[:68,1].max()
        w = x1 - x0
        h = y1 - y0
        w0_margin = w/8
        w1_margin = w/8
        h0_margin = h/2
        h1_margin = h/5
    
    if margin:
        w0_margin *= 4
        w1_margin *= 4
        h0_margin *= 2
        h1_margin *= 2
    elif phase == 'train':
        rand_factor = np.random.rand(4) * 0.6 + 0.2
        w0_margin *= rand_factor[0]
        w1_margin *= rand_factor[1]
        h0_margin *= rand_factor[2]
        h1_margin *= rand_factor[3]
    else:
        w0_margin *= 0.5
        w1_margin *= 0.5
        h0_margin *= 0.5
        h1_margin *= 0.5
            
    y0_new = max(0, int(y0-h0_margin))
    y1_new = min(H, int(y1+h1_margin)+1)
    x0_new = max(0, int(x0-w0_margin))
    x1_new = min(W, int(x1+w1_margin)+1)
    
    img_cropped = img[y0_new:y1_new, x0_new:x1_new]
    
    if only_img:
        return img_cropped
        
    if landmark is not None:
        landmark_cropped = np.zeros_like(landmark)
        for i, (p,q) in enumerate(landmark):
            landmark_cropped[i] = [p-x0_new, q-y0_new]
    else:
        landmark_cropped = None
        
    if bbox is not None:
        bbox_cropped = np.zeros_like(bbox)
        for i, (p,q) in enumerate(bbox):
            bbox_cropped[i] = [p-x0_new, q-y0_new]
    else:
        bbox_cropped = None

    if abs_coord:
        return img_cropped, landmark_cropped, bbox_cropped, (y0-y0_new, x0-x0_new, y1_new-y1, x1_new-x1), y0_new, y1_new, x0_new, x1_new
    else:
        return img_cropped, landmark_cropped, bbox_cropped, (y0-y0_new, x0-x0_new, y1_new-y1, x1_new-x1)