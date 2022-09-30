import os
import time
import cv2

import torch
import numpy as np
from tkinter import messagebox

from interactive_demo.brush import Brushstroke
from isegm.inference import clicker
from isegm.inference.predictors import get_predictor
from isegm.utils.vis import draw_with_blend_and_clicks


class InteractiveController:
    def __init__(self, net, device, predictor_params, update_image_callback, prob_thresh=0.5):
        self.net = net
        self.prob_thresh = prob_thresh
        self.clicker = clicker.Clicker()
        self.brush_stroke = None

        self.states = []
        self.probs_history = []
        self.object_count = 0
        self._result_mask = None
        self._init_mask = None

        self.image = None
        self.predictor = None
        self.device = device
        self.update_image_callback = update_image_callback
        self.predictor_params = predictor_params
        self.reset_predictor()


    def set_image(self, image):
        self.image = image
        self._result_mask = np.zeros(image.shape[:2], dtype=np.uint16)
        self.object_count = 0
        self.reset_last_object(update_image=False)
        self.update_image_callback(reset_canvas=True)

    def set_mask(self, mask):
        if self.image.shape[:2] != mask.shape[:2]:
            messagebox.showwarning("Warning", "A segmentation mask must have the same sizes as the current image!")
            return

        if len(self.probs_history) > 0:
            self.reset_last_object()

        self._init_mask = mask.astype(np.float32)
        self.probs_history.append((np.zeros_like(self._init_mask), self._init_mask))
        self._init_mask = torch.tensor(self._init_mask, device=self.device).unsqueeze(0).unsqueeze(0)
        self.clicker.click_indx_offset = 1

    def add_click(self, x, y, is_positive):
        self.states.append({
            'clicker': self.clicker.get_state(),
            'predictor': self.predictor.get_states(),
        })

        click = clicker.Click(is_positive=is_positive, coords=(y, x))
        self.clicker.add_click(click)
        pred = self.predictor.get_prediction(self.clicker, prev_mask=self._init_mask)
        if self._init_mask is not None and len(self.clicker) == 1:
            pred = self.predictor.get_prediction(self.clicker, prev_mask=self._init_mask)

        torch.cuda.empty_cache()

        if self.probs_history:
            self.probs_history.append((self.probs_history[-1][0], pred))
        else:
            self.probs_history.append((np.zeros_like(pred), pred))

        self.update_image_callback()

    def draw_brush(self, x, y, is_positive, radius=20):
        new_p = 1.0 if is_positive else 0.0
        is_new_brush_stroke = self.brush_stroke is None
        if is_new_brush_stroke:
            self.brush_stroke = Brushstroke(new_p, radius, self.image.shape[:2])

        self.brush_stroke.add_point((x, y))
        new_brush_points = self.brush_stroke.get_new_brush_points()
        if len(new_brush_points) == 0:
            return

        bound_x_1 = max(0, np.min(new_brush_points[:, 0]) - radius)
        bound_x_2 = min(self.image.shape[1], np.max(new_brush_points[:, 0]) + radius) + 1
        bound_y_1 = max(0, np.min(new_brush_points[:, 1]) - radius)
        bound_y_2 = min(self.image.shape[0], np.max(new_brush_points[:, 1]) + radius) + 1
        bounded_update_area = dict(x1 = bound_x_1, x2 = bound_x_2, y1 = bound_y_1, y2 = bound_y_2)

        pred = self.predictor.update_prediction(new_brush_points, radius, new_p)

        torch.cuda.empty_cache()

        if not self.probs_history:
            self.probs_history.append((np.zeros_like(pred), pred))
        elif is_new_brush_stroke:
            self.probs_history.append((self.probs_history[-1][0], pred))
        else:
            self.probs_history[-1] = (self.probs_history[-1][0], pred)

        self.update_image_callback(bounded_update_area=bounded_update_area)


    def end_brush_stroke(self):
        self.states.append({
            'clicker': self.clicker.get_state(),
            'predictor': self.predictor.get_states(),
        })

        torch.cuda.empty_cache()

        self.brush_stroke = None

    def undo(self):
        if not self.states:
            return

        prev_state = self.states.pop()
        self.clicker.set_state(prev_state['clicker'])
        self.predictor.set_states(prev_state['predictor'])
        self.probs_history.pop()
        if not self.probs_history:
            self.reset_init_mask()
        self.update_image_callback()

    def partially_finish_object(self):
        object_prob = self.current_object_prob
        if object_prob is None:
            return

        self.probs_history.append((object_prob, np.zeros_like(object_prob)))
        self.states.append(self.states[-1])

        self.clicker.reset_clicks()
        self.reset_predictor()
        self.reset_init_mask()
        self.update_image_callback()

    def finish_object(self):
        if self.current_object_prob is None:
            return

        self._result_mask = self.result_mask
        self.object_count += 1
        self.reset_last_object()

    def reset_last_object(self, update_image=True):
        self.states = []
        self.probs_history = []
        self.clicker.reset_clicks()
        self.reset_predictor()
        self.reset_init_mask()
        if update_image:
            self.update_image_callback()

    def reset_predictor(self, predictor_params=None):
        if predictor_params is not None:
            self.predictor_params = predictor_params
        self.predictor = get_predictor(self.net, device=self.device,
                                       **self.predictor_params)
        if self.image is not None:
            self.predictor.set_input_image(self.image)

    def reset_init_mask(self):
        self._init_mask = None
        self.clicker.click_indx_offset = 0

    @property
    def current_object_prob(self):
        if self.probs_history:
            current_prob_total, current_prob_additive = self.probs_history[-1]
            return np.maximum(current_prob_total, current_prob_additive)
        else:
            return None

    @property
    def is_incomplete_mask(self):
        return len(self.probs_history) > 0

    @property
    def result_mask(self):
        result_mask = self._result_mask.copy()
        if self.probs_history:
            result_mask[self.current_object_prob > self.prob_thresh] = self.object_count + 1
        return result_mask

    def get_visualization(self, alpha_blend, click_radius, canvas_img=None,
                          bounded_update_area=None):
        if self.image is None:
            return None

        results_mask_for_vis = self.result_mask

        vis = draw_with_blend_and_clicks(self.image, mask=results_mask_for_vis, alpha=alpha_blend,
                                         clicks_list=self.clicker.clicks_list, radius=click_radius,
                                         canvas_img=canvas_img, bound_area=bounded_update_area)

        return vis
