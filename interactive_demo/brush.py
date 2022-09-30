import numpy as np
from numpy.polynomial import Polynomial as P

class Brushstroke:
    """Represents a continuous brushstroke on the image while the user is drawing."""
    def __init__(self, probability, radius, img_shape):
        self.probability = probability
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
        self._prev_coords = self._coords
        self._coords = coords

    def get_new_brush_points(self):
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
        for i, (cur_y, cur_x) in enumerate(vertices[:-1]):
            cur_y, cur_x = int(round(cur_y)), int(round(cur_x))
            next_y, next_x = np.round(vertices[i + 1]).astype(int)
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
        vertices = np.round(vertices).astype(int)

        return vertices
