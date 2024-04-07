import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)   
    
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image




def make_line_points(y1, y2, line):

    if line is None:  
        return None
    slope, intercept = line 
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return ((x1, int(y1)), (x2, int(y2)))



def draw_lines(img, lines, color=[0, 0, 255], thickness=10):
    left_lines = [] 
    right_lines = []  

    for line in lines:
        for x1, y1, x2, y2 in line:
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if 0.4 < abs(slope) < 5:
                if slope < 0:
                    left_lines.append((slope, intercept))
                else:
                    right_lines.append((slope, intercept))
    
    if left_lines and right_lines:
        left_lane = np.mean(left_lines, axis=0)
        right_lane = np.mean(right_lines, axis=0)
        
        bottom_y = img.shape[0]
        mid_y = int(img.shape[0] * 0.65)
        
        left_bottom_x = int((bottom_y - left_lane[1]) / left_lane[0])
        right_bottom_x = int((bottom_y - right_lane[1]) / right_lane[0])
        
        left_mid_x = int((mid_y - left_lane[1]) / left_lane[0])
        right_mid_x = int((mid_y - right_lane[1]) / right_lane[0])
        
        mid_start_x = (left_bottom_x + right_bottom_x) // 2
        mid_end_x = (left_mid_x + right_mid_x) // 2
        
        cv2.line(img, (left_bottom_x, bottom_y), (left_mid_x, mid_y), color, thickness)
        cv2.line(img, (right_bottom_x, bottom_y), (right_mid_x, mid_y), color, thickness)
        
        cv2.line(img, (mid_start_x, bottom_y), (mid_end_x, mid_y), [0, 255, 0], thickness)




def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):

    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):

    return cv2.addWeighted(initial_img, α, img, β, γ)


def convert_to_hsv_and_filter(img, filter_yellow=True, filter_white=True):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    if filter_yellow:
        lower_yellow = np.array([50, 100, 100])
        upper_yellow = np.array([255, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    else:
        yellow_mask = 0

    if filter_white:
        lower_white = np.array([0, 0, 0])
        upper_white = np.array([255, 255, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
    else:
        white_mask = 0

    mask = cv2.bitwise_or(yellow_mask, white_mask)
    filtered_img = cv2.bitwise_and(img, img, mask=mask)
    return filtered_img


cap = cv2.VideoCapture('test_videos/bonus.mp4')

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('./method2_bonus.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    # gray = grayscale(frame)
    hsv_filtered = convert_to_hsv_and_filter(frame)
    
    blur_gray = gaussian_blur(hsv_filtered, 3)
    
    edges = canny(blur_gray, 50, 400)
    
    imshape = frame.shape
    vertices = np.array([[(0,imshape[0]),(750, 670), (1000, 670), (imshape[1],imshape[0])]], dtype=np.int32)

    masked_edges = region_of_interest(edges, vertices)
    
    
    lines = hough_lines(masked_edges, 1, np.pi/30, 50, 2, 3)
    
    lines_edges = weighted_img(lines, frame, 0.8, 1, 0)

    out.write(lines_edges)

    masked_edges_colored = cv2.cvtColor(masked_edges, cv2.COLOR_GRAY2BGR)
    
    combined = np.hstack((masked_edges_colored, lines_edges))
    
    cv2.imshow('Lane Lines', combined)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()