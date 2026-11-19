import numpy as np
import open_clip
class CLIPModel:
    def __init__(self,device,model_name="ViT-H-14",pretrained="laion2b_s32b_b79k"):
        self.clip_model,_,self.clip_preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained
        )
        self.clip_model = self.clip_model.to(device)
        self.clip_tokenizer = open_clip.get_tokenizer(model_name)
        self.device = device
        self.target_feat = None
    def compute_clip_features(self,image,results):
        
        padding = 20  # Adjust the padding amount as needed
        
        image_crops = []
        image_feats = []
        text_feats = []

        
        for idx in range(len(results['xyxy'])):
            # Get the crop of the mask with padding
            x_min, y_min, x_max, y_max = results['xyxy'][idx]

            # Check and adjust padding to avoid going beyond the image borders
            image_width, image_height = image.size
            left_padding = min(padding, x_min)
            top_padding = min(padding, y_min)
            right_padding = min(padding, image_width - x_max)
            bottom_padding = min(padding, image_height - y_max)

            # Apply the adjusted padding
            x_min -= left_padding
            y_min -= top_padding
            x_max += right_padding
            y_max += bottom_padding

            cropped_image = image.crop((x_min, y_min, x_max, y_max))
            
            # Get the preprocessed image for clip from the crop 
            preprocessed_image = self.clip_preprocess(cropped_image).unsqueeze(0).to(self.device)

            crop_feat = self.clip_model.encode_image(preprocessed_image)
            crop_feat /= crop_feat.norm(dim=-1, keepdim=True)
            
            tokenized_text = self.clip_tokenizer([results['classes'][idx]]).to(self.device)
            text_feat = self.clip_model.encode_text(tokenized_text)
            text_feat /= text_feat.norm(dim=-1, keepdim=True)
            
            crop_feat = crop_feat.detach().cpu().numpy()
            text_feat = text_feat.detach().cpu().numpy()

            image_crops.append(cropped_image)
            image_feats.append(crop_feat)
            text_feats.append(text_feat)
            
        # turn the list of feats into np matrices
        image_feats = np.concatenate(image_feats, axis=0)
        text_feats = np.concatenate(text_feats, axis=0)
        results['image_feats'] = image_feats
        results['text_feats'] = text_feats  
        return results
    def target_feature(self,text):
        text = self.clip_tokenizer([text]).to(self.device)
        text2_feat = self.clip_model.encode_text(text)
        text2_feat /= text2_feat.norm(dim=-1, keepdim=True)
        self.target_feat= text2_feat
        return text2_feat
