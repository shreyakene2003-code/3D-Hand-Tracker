import cv2
import mediapipe as mp
import numpy as np
import math

def get_rotation_matrix(angle_x, angle_y, angle_z):
    rx = np.array([
        [1, 0, 0],
        [0, math.cos(angle_x), -math.sin(angle_x)],
        [0, math.sin(angle_x), math.cos(angle_x)]
    ])
    ry = np.array([
        [math.cos(angle_y), 0, math.sin(angle_y)],
        [0, 1, 0],
        [-math.sin(angle_y), 0, math.cos(angle_y)]
    ])
    rz = np.array([
        [math.cos(angle_z), -math.sin(angle_z), 0],
        [math.sin(angle_z), math.cos(angle_z), 0],
        [0, 0, 1]
    ])
    return np.dot(rz, np.dot(ry, rx))

def project_3d(point_3d, center, scale, rot_mat):
    rotated_point = np.dot(rot_mat, point_3d)
    x_2d = int(rotated_point[0] * scale + center[0])
    y_2d = int(rotated_point[1] * scale + center[1])
    return (x_2d, y_2d)

def draw_cube(img, center, scale, angle_x, angle_y, angle_z, color=(0, 0, 0)):
    rot_mat = get_rotation_matrix(angle_x, angle_y, angle_z)
    grid_size = 9 # more boxes (dense grid)
    line_thickness = 4 # bold lines
    for d1 in np.linspace(-1, 1, grid_size):
        for d2 in np.linspace(-1, 1, grid_size):
            p1 = project_3d(np.array([-1, d1, d2]), center, scale, rot_mat)
            p2 = project_3d(np.array([1, d1, d2]), center, scale, rot_mat)
            cv2.line(img, p1, p2, color, line_thickness)
            p1 = project_3d(np.array([d1, -1, d2]), center, scale, rot_mat)
            p2 = project_3d(np.array([d1, 1, d2]), center, scale, rot_mat)
            cv2.line(img, p1, p2, color, line_thickness)
            p1 = project_3d(np.array([d1, d2, -1]), center, scale, rot_mat)
            p2 = project_3d(np.array([d1, d2, 1]), center, scale, rot_mat)
            cv2.line(img, p1, p2, color, line_thickness)

def draw_pyramid(img, center, scale, angle_x, angle_y, angle_z, color=(0, 0, 0)):
    rot_mat = get_rotation_matrix(angle_x, angle_y, angle_z)
    grid_size = 9 # more boxes
    line_thickness = 4 # bold lines
    stepper = np.linspace(-1, 1, grid_size)
    for ds in np.linspace(0, 1, grid_size):
        y = 1 - 2*ds
        s = 1 - ds
        for d in np.linspace(-s, s, grid_size):
            p1 = project_3d(np.array([-s, y, d]), center, scale, rot_mat)
            p2 = project_3d(np.array([s, y, d]), center, scale, rot_mat)
            cv2.line(img, p1, p2, color, line_thickness)
            p1 = project_3d(np.array([d, y, -s]), center, scale, rot_mat)
            p2 = project_3d(np.array([d, y, s]), center, scale, rot_mat)
            cv2.line(img, p1, p2, color, line_thickness)
    for dx in stepper:
        for dz in stepper:
            if dx in [-1, 1] or dz in [-1, 1]:
                p1 = project_3d(np.array([dx, 1, dz]), center, scale, rot_mat)
                p2 = project_3d(np.array([0, -1, 0]), center, scale, rot_mat)
                cv2.line(img, p1, p2, color, line_thickness)

def draw_sphere(img, center, scale, angle_x, angle_y, angle_z, color=(0, 0, 0)):
    rot_mat = get_rotation_matrix(angle_x, angle_y, angle_z)
    line_thickness = 4 # bold lines
    cv2.circle(img, (int(center[0]), int(center[1])), int(scale), color, line_thickness)
    
    lats = 12 # more dense grid
    longs = 20
    for i in range(1, lats):
        theta = (i / lats) * math.pi
        y = math.cos(theta)
        r = math.sin(theta)
        for j in range(longs):
            phi1 = (j / longs) * 2 * math.pi
            phi2 = ((j + 1) / longs) * 2 * math.pi
            p1 = project_3d(np.array([r * math.cos(phi1), y, r * math.sin(phi1)]), center, scale, rot_mat)
            p2 = project_3d(np.array([r * math.cos(phi2), y, r * math.sin(phi2)]), center, scale, rot_mat)
            cv2.line(img, p1, p2, color, line_thickness)
            
    for j in range(longs):
        phi = (j / longs) * 2 * math.pi
        for i in range(lats * 2):
            theta1 = (i / (lats * 2)) * math.pi
            theta2 = ((i + 1) / (lats * 2)) * math.pi
            p1 = project_3d(np.array([math.sin(theta1) * math.cos(phi), math.cos(theta1), math.sin(theta1) * math.sin(phi)]), center, scale, rot_mat)
            p2 = project_3d(np.array([math.sin(theta2) * math.cos(phi), math.cos(theta2), math.sin(theta2) * math.sin(phi)]), center, scale, rot_mat)
            cv2.line(img, p1, p2, color, line_thickness)

def draw_3d_grid(img):
    h, w, _ = img.shape
    center_x, center_y = w // 2, int(h * 0.3)
    color = (60, 60, 60)
    for i in range(-15, 16):
        x_bottom = int(center_x + i * 150)
        cv2.line(img, (center_x, center_y), (x_bottom, h), color, 1)
        
    for i in range(1, 15):
        y = int(center_y + (h - center_y) * (i / 15.0)**2)
        cv2.line(img, (0, y), (w, y), color, 1)

class ShapeState:
    def __init__(self, x, y):
        self.pos = [x, y]
        self.scale = 100
        self.angle_x = 0.0
        self.angle_y = 0.0
        self.angle_z = 0.0
        self.type = 'CUBE'
        self.color = (0, 0, 0)
        self.grabbed = False

def main():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2)
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(1)
    shape = None

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image = cv2.flip(image, 1)
        h, w, c = image.shape
        
        if shape is None:
            shape = ShapeState(w // 2, h // 2)

        # Draw 3D floor grid
        draw_3d_grid(image)

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_image)

        # Draw UI
        cv2.rectangle(image, (20, 20), (120, 80), (0, 255, 0), -1 if shape.type == 'CUBE' else 2)
        cv2.putText(image, "SQUARE", (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0) if shape.type == 'CUBE' else (0, 255, 0), 2)
        
        cv2.rectangle(image, (140, 20), (280, 80), (0, 255, 255), -1 if shape.type == 'PYRAMID' else 2)
        cv2.putText(image, "TRIANGLE", (150, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0) if shape.type == 'PYRAMID' else (0, 255, 255), 2)
        
        cv2.rectangle(image, (300, 20), (440, 80), (255, 150, 0), -1 if shape.type == 'SPHERE' else 2)
        cv2.putText(image, "CIRCLE", (315, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0) if shape.type == 'SPHERE' else (255, 150, 0), 2)
        
        # High Tech UI Text
        cv2.putText(image, "1 HAND : Pts (Thumb/Index) for Pos/Scale/Rot", (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(image, "2 HANDS: Global Scale/Rot between hands", (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        active_hands = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                
                tx, ty = int(thumb.x * w), int(thumb.y * h)
                ix, iy = int(index.x * w), int(index.y * h)
                
                cx, cy = (tx + ix) // 2, (ty + iy) // 2
                
                # Check UI Button Interactions (hover index tip)
                in_ui = False
                if iy > 20 and iy < 80:
                    if 20 < ix < 120: 
                        shape.type = 'CUBE'; shape.color = (0, 0, 0); in_ui = True
                    elif 140 < ix < 280: 
                        shape.type = 'PYRAMID'; shape.color = (0, 0, 0); in_ui = True
                    elif 300 < ix < 440: 
                        shape.type = 'SPHERE'; shape.color = (0, 0, 0); in_ui = True
                
                if not in_ui:
                    active_hands.append({
                        'cx': cx, 'cy': cy,
                        'tx': tx, 'ty': ty,
                        'ix': ix, 'iy': iy
                    })

        # High Tech Fluid Interaction Logic
        if len(active_hands) == 1:
            h1 = active_hands[0]
            shape.grabbed = True
            
            # Smooth tracking position
            shape.pos[0] += (h1['cx'] - shape.pos[0]) * 0.4
            shape.pos[1] += (h1['cy'] - shape.pos[1]) * 0.4
            
            # 1 Hand Scale (distance between thumb and index)
            dist = math.hypot(h1['ix'] - h1['tx'], h1['iy'] - h1['ty'])
            target_scale = max(30, dist * 1.5)
            shape.scale += (target_scale - shape.scale) * 0.3
            
            # 1 Hand Rotation
            angle_z = math.atan2(h1['iy'] - h1['ty'], h1['ix'] - h1['tx'])
            diff = angle_z - shape.angle_z
            diff = (diff + math.pi) % (2 * math.pi) - math.pi
            shape.angle_z += diff * 0.2
            
            target_angle_x = ((h1['cy'] - h/2) / (h/2)) * math.pi
            target_angle_y = ((h1['cx'] - w/2) / (w/2)) * math.pi
            shape.angle_x += (target_angle_x - shape.angle_x) * 0.2
            shape.angle_y += (target_angle_y - shape.angle_y) * 0.2
            
            # High Tech Tracking HUD
            cv2.line(image, (h1['tx'], h1['ty']), (h1['ix'], h1['iy']), (0, 255, 0), 2)
            cv2.circle(image, (h1['tx'], h1['ty']), 8, (0, 255, 255), -1)
            cv2.circle(image, (h1['ix'], h1['iy']), 8, (0, 255, 255), -1)
            cv2.circle(image, (int(shape.pos[0]), int(shape.pos[1])), int(shape.scale) + 20, (0, 255, 0), 1)
            
        elif len(active_hands) >= 2:
            h1, h2 = active_hands[0], active_hands[1]
            shape.grabbed = True
            
            mid_x = (h1['cx'] + h2['cx']) / 2
            mid_y = (h1['cy'] + h2['cy']) / 2
            
            shape.pos[0] += (mid_x - shape.pos[0]) * 0.4
            shape.pos[1] += (mid_y - shape.pos[1]) * 0.4
            
            dist_hands = math.hypot(h2['cx'] - h1['cx'], h2['cy'] - h1['cy'])
            target_scale = max(40, dist_hands / 1.5)
            shape.scale += (target_scale - shape.scale) * 0.3
            
            angle_z = math.atan2(h2['cy'] - h1['cy'], h2['cx'] - h1['cx'])
            diff = angle_z - shape.angle_z
            diff = (diff + math.pi) % (2 * math.pi) - math.pi
            shape.angle_z += diff * 0.2
            
            target_angle_x = ((mid_y - h/2) / (h/2)) * math.pi
            target_angle_y = ((mid_x - w/2) / (w/2)) * math.pi
            shape.angle_x += (target_angle_x - shape.angle_x) * 0.2
            shape.angle_y += (target_angle_y - shape.angle_y) * 0.2
            
            # High Tech Tracking HUD
            cv2.line(image, (h1['cx'], h1['cy']), (h2['cx'], h2['cy']), (0, 255, 0), 3)
            cv2.circle(image, (int(mid_x), int(mid_y)), 10, (0, 0, 255), -1)
            cv2.circle(image, (int(shape.pos[0]), int(shape.pos[1])), int(shape.scale) + 30, (0, 255, 0), 2)
            
        else:
            shape.grabbed = False
            # Idol spin
            shape.angle_x += 0.02
            shape.angle_y += 0.03
            shape.angle_z += 0.005
        
        # Render Shape
        center_tuple = (int(shape.pos[0]), int(shape.pos[1]))
        if shape.type == 'CUBE':
            draw_cube(image, center_tuple, shape.scale, shape.angle_x, shape.angle_y, shape.angle_z, shape.color)
        elif shape.type == 'PYRAMID':
            draw_pyramid(image, center_tuple, shape.scale, shape.angle_x, shape.angle_y, shape.angle_z, shape.color)
        elif shape.type == 'SPHERE':
            draw_sphere(image, center_tuple, shape.scale, shape.angle_x, shape.angle_y, shape.angle_z, shape.color)

        cv2.imshow('3D Hand Tracker', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
