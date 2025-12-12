import cv2
import mediapipe as mp
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math
import random

# --- SETTINGS ---
WINDOW_WIDTH, WINDOW_HEIGHT = 1000, 800
NUM_PARTICLES = 1000
PARTICLE_SIZE = 5 

# --- MEDIAPIPE SETUP ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands_detector = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=2
)

# --- FLUID PARTICLE CLASS ---
class Particle:
    def __init__(self):
        self.x = random.uniform(-0.5, 0.5)
        self.y = random.uniform(-0.5, 0.5)
        self.z = random.uniform(-0.5, 0.5)
        self.vx, self.vy, self.vz = 0, 0, 0
        self.color = (0.0, random.uniform(0.5, 1.0), 1.0)

    def update(self, rotation_speed, cube_pitch, cube_yaw):
        g_force = 0.003 
        rad_pitch = math.radians(cube_pitch)
        rad_yaw = math.radians(cube_yaw)

        # 1. Gravity (Rotated)
        gy_temp = -g_force * math.cos(-rad_pitch)
        gz_temp = -g_force * math.sin(-rad_pitch)
        gx_local = gz_temp * math.sin(-rad_yaw)
        gy_local = gy_temp
        gz_local = gz_temp * math.cos(-rad_yaw)

        self.vx += gx_local
        self.vy += gy_local
        self.vz += gz_local

        # 2. Centrifugal Force
        swirl = rotation_speed * 0.08
        self.vx += self.x * swirl
        self.vz += self.z * swirl

        # 3. Piling (Anti-Clump)
        if abs(self.vx) < 0.005 and abs(self.vy) < 0.005:
            self.vx += random.uniform(-0.005, 0.005)
            self.vz += random.uniform(-0.005, 0.005)

        # Update
        self.x += self.vx
        self.y += self.vy
        self.z += self.vz
        
        # Friction & Bounce
        self.vx *= 0.96; self.vy *= 0.96; self.vz *= 0.96
        bounce = -0.5 
        boundary = 0.95
        
        if self.x > boundary: self.x = boundary; self.vx *= bounce
        elif self.x < -boundary: self.x = -boundary; self.vx *= bounce

        if self.y > boundary: self.y = boundary; self.vy *= bounce
        elif self.y < -boundary: 
            self.y = -boundary
            self.vy *= bounce
            self.vx += random.uniform(-0.02, 0.02)
            self.vz += random.uniform(-0.02, 0.02)
            
        if self.z > boundary: self.z = boundary; self.vz *= bounce
        elif self.z < -boundary: self.z = -boundary; self.vz *= bounce

    def draw(self):
        glColor3f(*self.color)
        glVertex3f(self.x, self.y, self.z)

# --- CORRECTED CUBE VERTICES ---
VERTICES = (
    (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, -1),
    (1, -1, 1), (1, 1, 1), (-1, 1, 1), (-1, -1, 1)
)
EDGES = ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))

def draw_cube_lines():
    glColor3f(1.0, 1.0, 1.0) 
    glLineWidth(3)
    glBegin(GL_LINES)
    for edge in EDGES:
        for vertex in edge:
            glVertex3f(*VERTICES[vertex])
    glEnd()

def main():
    pygame.init()
    display = (WINDOW_WIDTH, WINDOW_HEIGHT)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("AI Hand Controller")
    
    glEnable(GL_DEPTH_TEST)
    glMatrixMode(GL_PROJECTION); glLoadIdentity()
    gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW); glLoadIdentity()
    glTranslatef(0.0, 0.0, -6)

    cap = cv2.VideoCapture(0)
    particles = [Particle() for _ in range(NUM_PARTICLES)]

    cube_scale = 1.0
    rotation_x, rotation_y = 0, 0
    target_rotation_speed = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False

        success, image = cap.read()
        if success:
            image = cv2.flip(image, 1) 
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands_detector.process(image_rgb)
            
            # Reset hand data
            left_index_tip = None
            left_thumb_tip = None
            right_index_tip = None

            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    label = results.multi_handedness[idx].classification[0].label
                    
                    index_tip = hand_landmarks.landmark[8]
                    thumb_tip = hand_landmarks.landmark[4]
                    
                    if label == "Left":
                        left_index_tip = (index_tip.x, index_tip.y)
                        left_thumb_tip = (thumb_tip.x, thumb_tip.y)
                    elif label == "Right":
                        right_index_tip = (index_tip.x, index_tip.y)

            # --- GESTURE LOGIC ---
            
            # MODE 1: RESIZE (Only if Left Hand exists AND Right Hand is missing)
            if left_index_tip and left_thumb_tip and (right_index_tip is None):
                pinch_dist = math.sqrt((left_index_tip[0] - left_thumb_tip[0])**2 + 
                                       (left_index_tip[1] - left_thumb_tip[1])**2)
                new_scale = pinch_dist * 8 
                cube_scale = max(0.5, min(new_scale, 2.5))
                # Stop rotation when resizing
                target_rotation_speed = 0 

            # MODE 2: ROTATE (Only if Both Hands exist)
            elif left_index_tip and right_index_tip:
                hand_dist = math.sqrt((left_index_tip[0] - right_index_tip[0])**2 + 
                                      (left_index_tip[1] - right_index_tip[1])**2)
                target_rotation_speed = hand_dist * 5
                
                avg_y = (left_index_tip[1] + right_index_tip[1]) / 2
                rotation_x = (0.5 - avg_y) * 90

            cv2.imshow('Camera View', image)
            
        # --- RENDER ---
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glPushMatrix() 
        
        glScalef(cube_scale, cube_scale, cube_scale)
        glRotatef(rotation_x, 1, 0, 0) 
        rotation_y += target_rotation_speed
        glRotatef(rotation_y, 0, 1, 0)

        draw_cube_lines()

        glPointSize(PARTICLE_SIZE)
        glBegin(GL_POINTS)
        for p in particles:
            p.update(target_rotation_speed, rotation_x, rotation_y)
            p.draw()
        glEnd()

        glPopMatrix()
        pygame.display.flip()
        pygame.time.wait(10)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()

if __name__ == "__main__":
    main()