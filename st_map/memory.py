"""Temporal memory methods extracted from the pinned ST-CogNav mapper.

No detector, simulator or GPU is needed to inspect/update existing memory records.
Objects use the original mapping dictionaries; semantic scoring is supplied by
``textmodel`` on the owning mapper. Frame captions remain opt-in.
"""
import os
from collections import Counter
import numpy as np
from .geometry import extractVoxelCenterfromVoxelmap, rotation_from_euler, X_w_to_X_v


def most_common_element(values):
    return Counter(values).most_common(1)[0][0] if values else None


class TemporalMemoryMixin:
    def generateCaption(self,obj,seg,img_path,use_vlm=False):
        name = most_common_element(obj['class_name']).replace('/','_')
        if seg['state'] == 0 :
            seg['caption'] = f"From timestep {seg['start_time']} to timestep {seg['end_time']}, instance {name} is not in view."
        elif seg['state'] == 1 :
            seg['caption'] = f"From timestep {seg['start_time']} to timestep {seg['end_time']}, instance {name} appears in view."
            if use_vlm:
                ###infer vlm to generate caption, at most 3 frames where the object was detected
                frames = sorted(set(i for i in obj['image_idx'] if seg['start_time'] <= i <= seg['end_time']))
                frames = [frames[i] for i in np.linspace(0,len(frames)-1,min(3,len(frames))).astype(int)] if len(frames) > 0 else []
                img_list = [os.path.join(img_path, str(i)+'.jpg') for i in frames if os.path.isfile(os.path.join(img_path, str(i)+'.jpg'))]
                if len(img_list) > 0:
                    print("infer vlm to generate caption")
                    description = (self.caption_vlm or self.vlm).query_multi_images(img_list,name)
                    if description is not None:
                        seg['caption'] += f" {description}"
        else :
            seg['caption'] = f"At timestep {seg['start_time']}, instance {name} disappears from its recorded location."
        return seg

    def updateStateSegment(self,obj,idx):
        if obj['caption'][-1]['state'] != obj['state_dict'][idx] :
            obj['caption'][-1]['end_time'] = idx -1
            obj['caption'].append({'start_time':idx,'end_time':None,'state':obj['state_dict'][idx],'caption':None})

    def appearanceMismatch(self,obj,conflict_uv,rgb):
        import cv2
        # compare the HS colour histogram of the object's latest detection mask with the current pixels it projects to
        if rgb is None or len(conflict_uv[0]) == 0:
            return True
        ref_img = cv2.imread(obj['color_path'][-1])
        ref_mask = np.asarray(obj['mask'][-1]).astype(np.uint8)
        if ref_img is None or ref_mask.shape != ref_img.shape[:2] or ref_mask.sum() == 0:
            return True
        cur_mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
        cur_mask[conflict_uv[1],conflict_uv[0]] = 1
        ref_hist = cv2.calcHist([cv2.cvtColor(ref_img,cv2.COLOR_BGR2HSV)],[0,1],ref_mask,[16,16],[0,180,0,256])
        cur_hist = cv2.calcHist([cv2.cvtColor(rgb.astype(np.uint8),cv2.COLOR_RGB2HSV)],[0,1],cur_mask,[16,16],[0,180,0,256])
        sim = cv2.compareHist(cv2.normalize(ref_hist,None),cv2.normalize(cur_hist,None),cv2.HISTCMP_CORREL)
        return sim < self.appearance_sim_thresh

    def archiveObjects(self,ids,idx):
        # remove state-2 nodes from the active graph and keep them in the history archive
        if len(ids) == 0:
            return
        for i in ids:
            obj = self.objects[i]
            obj['exists_in_SG'] = False
            obj['caption'][-1]['end_time'] = idx
            obj['archived_time'] = idx
            self.history_archive.append(obj)
            print("archive object:",most_common_element(obj['class_name']),"step:",idx)
        keep = [i for i in range(len(self.objects)) if i not in ids]
        if len(self.target_similarity_with_objects) == len(self.objects):
            self.target_similarity_with_objects = [self.target_similarity_with_objects[i] for i in keep]
        self.objects = type(self.objects)([self.objects[i] for i in keep])

    # per-detection lists that stay index-aligned with image_idx
    VIEW_KEYS = ('image_idx','mask_idx','color_path','mask','xyxy','n_points','pixel_area','contain_number')
    # what an archived object still needs: its name, where it was, its state history and a few views
    ARCHIVE_KEYS = ('class_name','class_id','voxel_index','caption','archived_time','state_dict','exists_in_SG',
                    'image_idx','xyxy','color_path','num_detections')

    def trimMemory(self,idx,save_path):
        """Keeps memory and disk use bounded over an episode of any length.

        Per object: the views_per_object detections with the largest pixel area (plus the latest, which
        appearanceMismatch compares against), the last class_votes labels, the last segments_per_object
        state segments, and in state_dict only the current step and the last step it was visible.
        The archive keeps the archive_limit most recent objects without point clouds, masks or features.
        Frames and poses no object refers to any more are deleted."""
        k = max(1,getattr(self,'views_per_object',3))
        votes = getattr(self,'class_votes',20)
        segments = getattr(self,'segments_per_object',8)
        archive_limit = getattr(self,'archive_limit',32)
        for obj in self.objects:
            states = obj['state_dict']
            if len(states) > 2:
                # the current state, and the last step the object was visible (shown to the planner)
                seen = max((t for t,v in states.items() if v == 1),default=None)
                obj['state_dict'] = {t: states[t] for t in sorted({seen,max(states)} - {None})}
            n = len(obj.get('image_idx',[]))
            if n > k and len(obj.get('pixel_area',[])) == n:
                area = obj['pixel_area']
                keep = sorted(set(sorted(range(n-1),key=lambda i:-float(area[i]))[:k-1]) | {n-1})
                for key in self.VIEW_KEYS:
                    if key in obj and isinstance(obj[key],list) and len(obj[key]) == n:
                        obj[key] = [obj[key][i] for i in keep]
            for key in ('class_name','class_id'):
                if key in obj and len(obj[key]) > votes:
                    obj[key] = obj[key][-votes:]
            if len(obj['caption']) > segments:
                obj['caption'] = obj['caption'][-segments:]
        for i,obj in enumerate(self.history_archive):
            if 'pcd' in obj or len(obj['caption']) > segments:
                slim = {key: obj[key] for key in self.ARCHIVE_KEYS if key in obj}
                slim['caption'] = slim['caption'][-segments:]
                self.history_archive[i] = slim
        if len(self.history_archive) > archive_limit:
            self.history_archive = self.history_archive[-archive_limit:]
        used = {idx} | {t for obj in self.objects for t in obj.get('image_idx',[])} | \
               {t for obj in self.history_archive for t in obj.get('image_idx',[])}
        for t in list(getattr(self,'agent_history',{}).keys()):
            if t not in used:
                del self.agent_history[t]
        for t in list(getattr(self,'saved_frames',())):
            if t not in used:
                path = os.path.join(save_path,'image',f'{t}.jpg')
                if os.path.exists(path):
                    os.remove(path)
                self.saved_frames.discard(t)

    def alignTargetSimilarity(self):
        """Recomputes target similarities when a merge or removal left the list out of step with the
        objects (target verification indexes both lists together)."""
        if len(self.target_similarity_with_objects) != len(self.objects):
            self.target_similarity_with_objects = [self.textmodel.query_multiple(o['class_name']) for o in self.objects]

    def retrieveHistoryContext(self,img_path,top_k=3,sim_thresh=0.5,use_vlm=None):
        # archived objects most related to the current instruction, with their state sequences
        if len(self.history_archive) == 0:
            return []
        if use_vlm is None:
            use_vlm = self.history_caption
        sims = [self.textmodel.query_multiple(obj['class_name'])[0] for obj in self.history_archive]
        order = [i for i in np.argsort(sims)[::-1] if sims[i] > sim_thresh][:top_k]
        context = []
        for i in order:
            obj = self.history_archive[i]
            segments = [seg for seg in obj['caption'] if seg['state'] != 0][-5:]
            for seg in segments:
                if seg['caption'] is None:
                    self.generateCaption(obj,seg,img_path,use_vlm)
            points = extractVoxelCenterfromVoxelmap(obj['voxel_index'],self.size,self.min_bound,0.25)
            context.append({
                'name': most_common_element(obj['class_name']).replace('/','_'),
                'similarity': sims[i],
                'center': np.mean(points[:,:2],axis=0),
                'archived_time': obj['archived_time'],
                'captions': [seg['caption'] for seg in segments],
            })
        return context
    def updateObjectState(self,idx,save_path,agent_locations,pcd,depth,rgb=None):
        R = rotation_from_euler(yaw=np.deg2rad((int((360 - self.agent_history[idx]['angle']+90) % 360)-90)%360), roll=np.deg2rad(-int(self.agent_history[idx]['roll'])))
        kernel = np.ones((15, 15), np.uint8)
        depth_buffer = depth
        mask = -np.ones((self.screen_h,self.screen_w), dtype=np.int32)
        ratios = {}
        visible_length = {}
        conflict_uv = {}
        for i,obj in enumerate(self.objects) :
            if obj is not None:
                points = np.array(obj['pcd'].points)
                X_w = X_w_to_X_v(R,[self.agent_history[idx]['pos']['x'],self.agent_history[idx]['pos']['y'],self.agent_history[idx]['pos']['z']],points)
                points_camera = self.env_camera_matrix[:3,:3] @ X_w
                points_camera = points_camera.T
                z = points_camera[:,-1]
                # divide only points in front of the near plane (others would give inf / NaN pixel coordinates)
                z_safe = np.where(z >= 0.1, z, 1.0)
                u = np.round(points_camera[:, 0] / z_safe).astype(np.int32)
                v = np.round(points_camera[:, 1] / z_safe).astype(np.int32)
                
                # Filter points inside the image plane and valid depth
                valid = (u >= 0) & (u < self.screen_w) & \
                        (v >= 0) & (v < self.screen_h) & \
                        (z >= 0.1)  # Near-clip check
                
                u_valid = u[valid]
                v_valid = v[valid]
                z_valid = z[valid]
                
                # Compare with depth buffer (only keep points closer than the rendered depth)
                depth_at_uv = depth_buffer[v_valid, u_valid]
                visible = np.where(z_valid <= (depth_at_uv + 0.01))[0]  # Allow slight tolerance (+1.0)
                # Update mask with object ID where visible
                if len(visible) > 0:
                    ratio = len(np.where((depth_at_uv[visible]-z_valid[visible])>0.1)[0])/len(visible)
                    visible_length[i] = len(visible)
                    ratios[i] = ratio
                    # pixels where the observed depth is behind the projected object (geometric conflict)
                    conflict = np.where((depth_at_uv[visible]-z_valid[visible])>0.1)[0]
                    conflict_uv[i] = (u_valid[visible][conflict], v_valid[visible][conflict])
                # Optional: Compute depth ratios for further analysis
        not_in_objects = list(set(np.arange(len(self.objects)))-set(list(visible_length.keys())))
        for i in not_in_objects :
            # objects merged with a detection at this step already have state 1
            if idx not in self.objects[i]['state_dict'].keys():
                self.objects[i]['state_dict'][idx] = 0
            self.updateStateSegment(self.objects[i],idx)
        missing = []
        for key,value in visible_length.items() :
                if idx not in self.objects[key]['state_dict'].keys():
                    if value > getattr(self,'min_visible_points',100):
                        # state 2 needs both geometric conflict and appearance mismatch
                        if ratios[key] > 0.2 and self.appearanceMismatch(self.objects[key],conflict_uv[key],rgb):
                            self.objects[key]['state_dict'][idx] = 2
                            missing.append(key)
                        else :
                            self.objects[key]['state_dict'][idx] = 1
                    else:
                        self.objects[key]['state_dict'][idx] = 0
                self.updateStateSegment(self.objects[key],idx)
        self.archiveObjects(missing,idx)
        self.trimMemory(idx,save_path)

