def convert_yolov8_bbox_to_opencv(bbox_xywh, image_width, image_height):
    """
    Convert YOLOv8 bounding box coordinates (x_center, y_center, width, height)
    to OpenCV bounding box coordinates (top-left, bottom-right).
    
    Args:
    - bbox_xywh: Tuple of (x_center, y_center, width, height)
    - image_width: Width of the image
    - image_height: Height of the image
    
    Returns:
    - opencv_bbox: Tuple of (x_top_left, y_top_left, x_bottom_right, y_bottom_right)
    """
    x_center, y_center, width, height = bbox_xywh
    x_center *= image_width
    y_center *= image_height
    width *= image_width
    height *= image_height
    
    x_top_left = int(x_center - width / 2)
    y_top_left = int(y_center - height / 2)
    x_bottom_right = int(x_center + width / 2)
    y_bottom_right = int(y_center + height / 2)
    
    return (x_top_left, y_top_left, x_bottom_right, y_bottom_right)
