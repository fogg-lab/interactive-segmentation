import numpy as np
from numpy.polynomial import Polynomial as P
import sys

class Brushstroke:
    """Represents a continuous brushstroke on the image while the user is drawing."""
    def __init__(self, probability, radius, img_shape):
        self.probability = probability
        self.radius = radius
        self.img_h, self.img_w = img_shape
        self._coords = None
        self._prev_coords = None
        self._fit_curve_p_left = None
        self._fit_curve_p_right = None
        self._fit_curve_extremal_point = None   # (x,y) of local minimum or maximum
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
        Fits a polynomial along brush points to estimate the brush stroke.

        Returns:
            numpy.ndarray: array of [x,y] vertices along the brush stroke.
        """

        if self._prev_coords is None:
            return np.array([[self._coords[0], self._coords[1]]]).astype(int)
        if self._prev_coords == self._coords:
            return np.array([]) # no new points along brushstroke

        fit_xy = np.zeros((3,), dtype=[('x', np.float64), ('y', np.float64)])
        fit_xy[1] = self._prev_coords[::-1] if self._fit_reflect_yx else self._prev_coords
        fit_xy[2] = self._coords[::-1] if self._fit_reflect_yx else self._coords

        if self._fit_curve_p_left is None or self._fit_curve_p_right is None:
            rank = 1
        else:
            # Add a point near the previous x along the last curve
            x0_x1_change = 0.0001
            fit_xy['x'][0] = fit_xy['x'][1] - x0_x1_change
            fit_xy['y'][0] = self._eval_poly(fit_xy['x'][0])

            if fit_xy['x'][1] == fit_xy['x'][2]:
                # Fit a quadratic between the 3 points only if the previous y is the midpoint y-val
                rank = 2 if fit_xy['y'][1] not in (np.min(fit_xy['y']), np.max(fit_xy['y'])) else 1
            elif np.all(fit_xy['y']==fit_xy['y'][0]):
                # Fit a horizontal line
                rank = 1
            else:
                rank = 2

            if rank == 2:
                # Fit a quadratic to the 3 points
                y0_y1_change = np.subtract(fit_xy['y'][1], fit_xy['y'][0])
                x1_tangent_slope_estimate = np.divide(y0_y1_change, x0_x1_change)
                x1_slope_sign = -1 if x1_tangent_slope_estimate < 0 else 1
                x1_min_slope_magnitude, x1_max_slope_magnitude = 0.01, 100

                if fit_xy['x'][1] == fit_xy['x'][2]:
                    x1_adjusted_slope_magnitude = max(np.absolute(x1_tangent_slope_estimate),
                                                    x1_min_slope_magnitude)
                else:
                    x1_adjusted_slope_magnitude = min(np.absolute(x1_tangent_slope_estimate),
                                                    x1_max_slope_magnitude)

                adjusted_y0_y1_change = x1_slope_sign * x0_x1_change * x1_adjusted_slope_magnitude
                fit_xy['y'][0] = np.subtract(fit_xy['y'][1], adjusted_y0_y1_change)

        if fit_xy['x'][1] == fit_xy['x'][2]:
            self._fit_reflect_yx = not self._fit_reflect_yx
            # Reflect x and y coordinates along y=x, so that (x,y) -> (y,x)
            fit_xy['x'], fit_xy['y'] = fit_xy['y'].copy(), fit_xy['x'].copy()

        # Prepare fit_xy to fit a new curve
        fit_xy = fit_xy[1:] if rank == 1 else fit_xy
        fit_xy = np.sort(fit_xy)

        # Fit the new curve
        self._fit_curve_p_left = P.fit(fit_xy['x'], fit_xy['y'], rank)
        self._fit_curve_p_right = self._fit_curve_p_left

        # Get the domain of the brush stroke (all x integer values between the two original points)
        coord_x_index = 1 if self._fit_reflect_yx else 0
        x_min, x_max = sorted((self._prev_coords[coord_x_index], self._coords[coord_x_index]))
        domain_space = np.linspace(start=x_min, stop=x_max, num=x_max-x_min+1)

        # Make sure the local min or max is reasonable
        x_at_min_max = [x for x in self._fit_curve_p_left.deriv().roots() if x_min<=x<=x_max]
        if len(x_at_min_max) == 1:
            self._fit_curve_extremal_point = (x_at_min_max, self._fit_curve_p_left(*x_at_min_max))
        else:
            self._fit_curve_extremal_point = None
        if self._fit_curve_extremal_point is not None:
            points_y_min, points_y_max = np.min(fit_xy['y']), np.max(fit_xy['y'])
            points_y_span = points_y_max - points_y_min
            curve_y_sorted = sorted((self._fit_curve_extremal_point[1], *fit_xy['y']))
            curve_y_min, curve_y_max = curve_y_sorted[0], curve_y_sorted[-1]
            curve_y_span = curve_y_max - curve_y_min
            curve_y_span_limit = points_y_span * 1.5 + 1
            compression_ratio = curve_y_span_limit / curve_y_span
            if compression_ratio < 1:
                min_x, max_x = fit_xy['x'][0], fit_xy['x'][-1]
                mid_x = self._fit_curve_extremal_point
                extremal_val = self._fit_curve_extremal_point[1]
                self._compress_poly_vertically(start_x=min_x, end_x=mid_x, mid_y=mid_y,
                                               extremal_val=extremal_val, ratio=compression_ratio,
                                               side="left")
                self._compress_poly_vertically(start_x=mid_x, end_x=max_x, mid_y=mid_y,
                                               extremal_val=extremal_val, ratio=compression_ratio,
                                               side="right")

        # Evaluate the polynomial at each x value in the domain to get an array of [y,x] vertices
        line_fitx = self._eval_poly(domain_space)

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
                    y_estimate = self._eval_poly(x_estimate)
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

    def _eval_poly(self, x):
        """Evaluate the polynomial at x."""
        """
        if self._fit_curve_extremal_point is not None and x < self._fit_curve_extremal_point[0]:
            result = self._fit_curve_p_left(x)
        else:
            result = self._fit_curve_p_right(x)
        return result
        """
        return self._fit_curve_p_left(x)

    def _compress_poly_vertically(self, start_x, end_x, mid_y, extremal_val, ratio, side):
        p = self._fit_curve_p_left.copy() if side=="left" else self._fit_curve_p_right.copy()
        compressor = np.zeros((3,), dtype=[('x', np.float64), ('y', np.float64)])

        compressor['x'] = [start_x, end_x, 2*end_x-start_x]
        midpoint_ratio = (mid_y * (1 - ratio) + extremal_val * ratio) / 2
        compressor['y'] = [1, midpoint_ratio, 1]
