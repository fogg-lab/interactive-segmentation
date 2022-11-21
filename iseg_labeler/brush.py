import cv2
import numpy as np
from numpy.polynomial import Polynomial as P


class Brush:
    def __init__(self, img_shape):
        self.current_brushstroke = None
        # Default value in brush mask is 2, which signifies it is not part of any brush stroke
        self._brush_mask = np.full(img_shape, 2, dtype=np.uint8)
        self.img_shape = img_shape
        self._bounds_of_last_brush_update = None

    def start_brushstroke(self, value: int, radius):
        """
        Start a new brushstroke.

        Args:
            value (int): Value of the brushstroke - 0 = background, 1 = tube, 2 = agnostic value.
            radius (int): The radius of the brushstroke.
        """
        self.current_brushstroke = Brushstroke(value, radius, self.img_shape)

    def add_brushstroke_point(self, coords):
        """
        Add a new point to the brushstroke.

        Args:
            coords (tuple): (x, y) coordinates of the new point.

        Returns:
            bool: True if brush mask was updated, False otherwise.
        """
        if self.current_brushstroke is None:
            raise ValueError("Current brushstroke is None. "
                             "You must call start_brushstroke() before adding points.")

        new_point_added = self.current_brushstroke.add_point(coords)
        if not new_point_added:
            return False    # Mask not updated (new point was same as the last one)

        new_points = self.current_brushstroke.estimate_new_brushstroke_points()

        if len(new_points > 0):
            self._update_mask(new_points)
            mask_updated = True
        else:
            mask_updated = False

        return mask_updated

    def get_brush_mask(self):
        return self._brush_mask, self._bounds_of_last_brush_update

    def end_brushstroke(self):
        self.current_brushstroke = None

    def _update_mask(self, points):
        """Update brush mask with filled in circles at given points."""

        radius = self.current_brushstroke.radius
        value = self.current_brushstroke.value

        # Add padding when circle is partially outside of image
        min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
        min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])

        top, bottom = max(0, radius - min_y), max(0, max_y + radius + 1 - self.img_shape[0])
        left, right = max(0, radius - min_x), max(0, max_x + radius + 1 - self.img_shape[1])
        self._brush_mask = cv2.copyMakeBorder(self._brush_mask, top, bottom, left, right,
                                              cv2.BORDER_CONSTANT, value=value)

        for x, y in points:
            cv2.circle(self._brush_mask, (x + left, y + top), radius, value, -1)

        # Remove padding that may have been added
        self._brush_mask = self._brush_mask[top:top+self.img_shape[0], left:left+self.img_shape[1]]

        # Update bounds of last brush update
        bound_x_1 = max(0, np.min(points[:, 0]) - radius)
        bound_x_2 = min(self.img_shape[1], np.max(points[:, 0]) + radius) + 1
        bound_y_1 = max(0, np.min(points[:, 1]) - radius)
        bound_y_2 = min(self.img_shape[0], np.max(points[:, 1]) + radius) + 1
        self._bounds_of_last_brush_update = dict(x1 = bound_x_1, x2 = bound_x_2,
                                                 y1 = bound_y_1, y2 = bound_y_2)


class Brushstroke:
    """Represents a continuous brushstroke on the image while the user is drawing."""
    def __init__(self, value: int, radius, img_shape):
        self.value = value
        self.vertices = None
        self.radius = radius
        self.img_h, self.img_w = img_shape
        self._coords = None
        self._prev_coords = None
        self._fit_line_p = None
        self._fit_reflect_yx = False

    def add_point(self, coords):
        """
        Add a new point to the brushstroke.

        Args:
            coords (tuple): (x, y) coordinates of the new point.
        """
        if coords == self._coords:
            return False    # Point is the same as the last one
        elif self._coords is not None:
            delta_x = abs(coords[0] - self._coords[0])
            delta_y = abs(coords[1] - self._coords[1])
            if delta_x > 100 or delta_y > 100:
                return False    # Point is too far from the last one
        self._prev_coords = self._coords
        self._coords = coords
        return True

    def estimate_new_brushstroke_points(self):
        """
        Fits a line along brush points to estimate the brush stroke.

        Returns:
            numpy.ndarray: array of [x,y] vertices along the brush stroke.
        """

        if self._prev_coords is None:
            return np.array([[*self._coords]])
        if self._prev_coords == self._coords:
            return np.array([]) # no new points along brushstroke

        fit_xy = np.zeros((2,), dtype=[('x', np.float64), ('y', np.float64)])
        fit_xy[0] = self._prev_coords[::-1] if self._fit_reflect_yx else self._prev_coords
        fit_xy[1] = self._coords[::-1] if self._fit_reflect_yx else self._coords

        if fit_xy['x'][0] == fit_xy['x'][1]:
            self._fit_reflect_yx = not self._fit_reflect_yx
            # Reflect x and y coordinates along y=x, so that (x,y) -> (y,x)
            fit_xy['x'], fit_xy['y'] = fit_xy['y'].copy(), fit_xy['x'].copy()

        # Prepare fit_xy to fit a new line
        self._fit_line_p = P.fit(fit_xy['x'], fit_xy['y'], deg=1)
        fit_xy = np.sort(fit_xy)
        x_min, x_max = int(fit_xy['x'][0]), int(fit_xy['x'][1])
        domain_space = np.linspace(start = x_min, stop = x_max, num = x_max - x_min + 1)
        line_fitx = self._fit_line_p(domain_space)
        vertices = np.array(list(zip(line_fitx, domain_space)))

        # Ensure all pixels along the brush stroke are accounted for in the vertices
        new_vertices = []
        for i, vertex in enumerate(vertices[:-1]):
            cur_y, cur_x = np.round(vertex).astype(np.int32)
            next_y, next_x = np.round(vertices[i + 1]).astype(np.int32)
            skipped_y_vals = np.arange(cur_y + 1, next_y)
            for _, skipped_y in enumerate(skipped_y_vals):
                x_left_bound, x_right_bound = cur_x, next_x
                x_estimate, y_estimate = 0, 0
                for _ in range(10):
                    x_estimate = (x_left_bound + x_right_bound) / 2
                    y_estimate = self._fit_line_p(x_estimate)
                    if y_estimate > skipped_y:
                        x_right_bound = x_estimate
                    elif y_estimate < skipped_y:
                        x_left_bound = x_estimate
                    else:
                        break
                new_vertices.append([y_estimate, x_estimate])

        # Add new vertices to the array of vertices
        if len(new_vertices) > 0:
            vertices = np.append(vertices, new_vertices, axis=0)

        # Ensure each row of vertices is in the form [x, y] relative to the image
        vertices = vertices if self._fit_reflect_yx else np.flip(vertices, 1)

        # Round vertices and convert to int
        vertices = np.round(vertices).astype(np.int32)

        # Add current vertices to total brushstroke vertices
        if self.vertices is None:
            self.vertices = vertices
        else:
            self.vertices = np.append(self.vertices, vertices, axis=0)

        return vertices
