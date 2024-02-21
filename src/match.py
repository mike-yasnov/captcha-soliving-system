import math

class BboxCircleMatcher:
    """
    Match orbitals with icons
    """
    def __init__(self, ):

        self.matched_pairs = {}


    def bbox_center(self, bbox: list):
        """
        Find bbox center 

        @param bbox: 
        """
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def circle_bbox_intersect(self, circle, bbox):
        bbox_center = self.bbox_center(bbox)
        circle_center = circle['center']
        radius = circle['radius']

        closest_x = bbox_center[0] if bbox_center[0] < circle_center[0] else max(bbox[0], min(circle_center[0], bbox[2]))
        closest_y = bbox_center[1] if bbox_center[1] < circle_center[1] else max(bbox[1], min(circle_center[1], bbox[3]))

        distance = math.sqrt((circle_center[0] - closest_x) ** 2 + (circle_center[1] - closest_y) ** 2)
        return distance < radius

    def match(self, circles, bboxes):
        for circle in circles:
            matched_bbox = None
            min_distance = float('inf')

            for bbox in bboxes:
                if self.circle_bbox_intersect(circle, bbox):
                    bbox_center = self.bbox_center(bbox)
                    circle_center = circle['center']
                    distance = math.sqrt((circle_center[0] - bbox_center[0]) ** 2 + (circle_center[1] - bbox_center[1]) ** 2)
                    if distance < min_distance:
                        min_distance = distance
                        matched_bbox = bbox

            self.matched_pairs[circle['id']] = matched_bbox

            if matched_bbox in bboxes:
                bboxes.remove(matched_bbox)

        unmatched_circles = [circle for circle, bbox in self.matched_pairs.items() if bbox is None]
        if len(unmatched_circles) == 1:
            self.matched_pairs[unmatched_circles[0]] = bboxes[-1]
            del bboxes[-1]

        return self.matched_pairs
