"""
30 DRONE 3D SİMÜLASYON - NESNE TESPİTİ VE KOORDİNELİ SALDIRI
===========================================================
Gerekli kütüphaneler:
pip install pygame pyopengl pyopengl_accelerate numpy
"""

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math
import random
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import threading
from collections import deque
import colorsys
from enum import Enum

# =====================================================
# NESNE TİPLERİ VE SINIFLAR
# =====================================================

class ObjectType(Enum):
    """Yer nesnesi tipleri"""
    CAR = "CAR"           # Araba - hedef
    BUILDING = "BUILDING" # Bina - engel
    TREE = "TREE"        # Ağaç - engel
    CIVILIAN = "CIVILIAN" # Sivil - kaçınılacak
    TRUCK = "TRUCK"      # Kamyon - ikincil hedef

class GroundObject:
    """Yer nesnesi sınıfı"""
    
    def __init__(self, obj_id: int, obj_type: ObjectType, position: np.ndarray):
        self.id = obj_id
        self.type = obj_type
        self.position = np.array(position, dtype=float)
        self.size = self._get_size_by_type()
        self.color = self._get_color_by_type()
        self.is_target = obj_type in [ObjectType.CAR, ObjectType.TRUCK]
        self.is_engaged = False  # Saldırı altında mı
        self.engaged_by = []  # Saldıran drone'lar
        self.destroyed = False
        self.health = 100.0 if self.is_target else float('inf')
        self.spawn_time = time.time()
        self.lifetime = random.uniform(20, 40) if self.is_target else float('inf')
        
        # Hareket özellikleri (arabalar için)
        if obj_type == ObjectType.CAR:
            self.velocity = np.array([random.uniform(-5, 5), random.uniform(-5, 5), 0])
            self.is_moving = True
        else:
            self.velocity = np.zeros(3)
            self.is_moving = False
    
    def _get_size_by_type(self) -> Tuple[float, float, float]:
        """Nesne tipine göre boyut"""
        sizes = {
            ObjectType.CAR: (4, 2, 1.5),
            ObjectType.BUILDING: (15, 15, 20),
            ObjectType.TREE: (2, 2, 8),
            ObjectType.CIVILIAN: (0.8, 0.8, 1.8),
            ObjectType.TRUCK: (7, 3, 3)
        }
        return sizes.get(self.type, (2, 2, 2))
    
    def _get_color_by_type(self) -> Tuple[float, float, float]:
        """Nesne tipine göre renk"""
        colors = {
            ObjectType.CAR: (1.0, 0.2, 0.2),      # Kırmızı
            ObjectType.BUILDING: (0.5, 0.5, 0.5),  # Gri
            ObjectType.TREE: (0.2, 0.8, 0.2),      # Yeşil
            ObjectType.CIVILIAN: (0.2, 0.2, 1.0),   # Mavi
            ObjectType.TRUCK: (0.8, 0.4, 0.2)      # Turuncu
        }
        return colors.get(self.type, (1, 1, 1))
    
    def update(self, dt: float, bounds: Tuple[float, float, float, float]):
        """Nesneyi güncelle"""
        if self.is_moving and not self.destroyed:
            # Pozisyonu güncelle
            self.position += self.velocity * dt
            
            # Sınırları kontrol et ve yansıt
            min_x, max_x, min_y, max_y = bounds
            if self.position[0] < min_x or self.position[0] > max_x:
                self.velocity[0] = -self.velocity[0]
                self.position[0] = np.clip(self.position[0], min_x, max_x)
            if self.position[1] < min_y or self.position[1] > max_y:
                self.velocity[1] = -self.velocity[1]
                self.position[1] = np.clip(self.position[1], min_y, max_y)
        
        # Yaşam süresi kontrolü
        if self.is_target and not self.destroyed:
            age = time.time() - self.spawn_time
            if age > self.lifetime:
                self.destroyed = True
    
    def take_damage(self, damage: float):
        """Hasar al"""
        if self.is_target and not self.destroyed:
            self.health -= damage
            if self.health <= 0:
                self.health = 0
                self.destroyed = True
                return True
        return False
    
    def draw(self):
        """Nesneyi çiz"""
        if self.destroyed:
            return
        
        glPushMatrix()
        glTranslatef(*self.position)
        
        # Saldırı altındaysa yanıp sönsün
        if self.is_engaged:
            pulse = abs(math.sin(time.time() * 5))
            glColor3f(
                self.color[0] * (0.5 + 0.5 * pulse),
                self.color[1] * (0.5 + 0.5 * pulse),
                self.color[2] * (0.5 + 0.5 * pulse)
            )
        else:
            glColor3f(*self.color)
        
        # Nesne tipine göre çizim
        if self.type == ObjectType.CAR:
            self._draw_car()
        elif self.type == ObjectType.BUILDING:
            self._draw_building()
        elif self.type == ObjectType.TREE:
            self._draw_tree()
        elif self.type == ObjectType.CIVILIAN:
            self._draw_civilian()
        elif self.type == ObjectType.TRUCK:
            self._draw_truck()
        
        glPopMatrix()
        
        # Hedef işareti
        if self.is_target and not self.destroyed:
            self._draw_target_marker()
    
    def _draw_car(self):
        """Araba çiz"""
        w, l, h = self.size
        
        # Gövde
        glBegin(GL_QUADS)
        # Üst
        glVertex3f(-w/2, -l/2, h)
        glVertex3f(w/2, -l/2, h)
        glVertex3f(w/2, l/2, h)
        glVertex3f(-w/2, l/2, h)
        # Yan yüzler
        for i in range(4):
            angle = i * 90
            x1 = w/2 * math.cos(math.radians(angle))
            y1 = l/2 * math.sin(math.radians(angle))
            x2 = w/2 * math.cos(math.radians(angle + 90))
            y2 = l/2 * math.sin(math.radians(angle + 90))
            glVertex3f(x1, y1, 0)
            glVertex3f(x2, y2, 0)
            glVertex3f(x2, y2, h)
            glVertex3f(x1, y1, h)
        glEnd()
        
        # Tekerlekler
        glColor3f(0.1, 0.1, 0.1)
        wheel_positions = [
            (-w/2 + 0.5, -l/2 + 0.5),
            (w/2 - 0.5, -l/2 + 0.5),
            (-w/2 + 0.5, l/2 - 0.5),
            (w/2 - 0.5, l/2 - 0.5)
        ]
        for wx, wy in wheel_positions:
            glPushMatrix()
            glTranslatef(wx, wy, 0.3)
            self._draw_cube(0.6, 0.3, 0.6)
            glPopMatrix()
    
    def _draw_building(self):
        """Bina çiz"""
        w, l, h = self.size
        self._draw_cube(w, l, h)
        
        # Pencereler
        glColor3f(0.8, 0.8, 0.2)
        for floor in range(1, int(h/3)):
            for window in range(3):
                glPushMatrix()
                glTranslatef(w/2 - 0.1, -l/2 + l/3 * window, floor * 3)
                self._draw_cube(0.2, 1, 1)
                glPopMatrix()
    
    def _draw_tree(self):
        """Ağaç çiz"""
        # Gövde
        glColor3f(0.4, 0.2, 0.1)
        self._draw_cube(0.5, 0.5, 3)
        
        # Yapraklar
        glColor3f(0.2, 0.8, 0.2)
        glPushMatrix()
        glTranslatef(0, 0, 4)
        self._draw_cube(3, 3, 4)
        glPopMatrix()
    
    def _draw_civilian(self):
        """Sivil çiz"""
        w, l, h = self.size
        self._draw_cube(w, l, h)
    
    def _draw_truck(self):
        """Kamyon çiz"""
        w, l, h = self.size
        self._draw_cube(w, l, h)
    
    def _draw_cube(self, width, length, height):
        """Basit küp çizimi"""
        w, l, h = width/2, length/2, height
        glBegin(GL_QUADS)
        # Ön
        glVertex3f(-w, -l, h)
        glVertex3f(w, -l, h)
        glVertex3f(w, -l, 0)
        glVertex3f(-w, -l, 0)
        # Arka
        glVertex3f(-w, l, h)
        glVertex3f(w, l, h)
        glVertex3f(w, l, 0)
        glVertex3f(-w, l, 0)
        # Sol
        glVertex3f(-w, -l, h)
        glVertex3f(-w, l, h)
        glVertex3f(-w, l, 0)
        glVertex3f(-w, -l, 0)
        # Sağ
        glVertex3f(w, -l, h)
        glVertex3f(w, l, h)
        glVertex3f(w, l, 0)
        glVertex3f(w, -l, 0)
        # Üst
        glVertex3f(-w, -l, h)
        glVertex3f(w, -l, h)
        glVertex3f(w, l, h)
        glVertex3f(-w, l, h)
        glEnd()
    
    def _draw_target_marker(self):
        """Hedef işareti çiz"""
        if self.is_engaged:
            glColor3f(1, 1, 0)  # Sarı
        else:
            glColor3f(1, 0, 0)  # Kırmızı
        
        glLineWidth(2)
        glBegin(GL_LINE_LOOP)
        radius = max(self.size) * 1.5
        for i in range(16):
            angle = i * math.pi * 2 / 16
            x = self.position[0] + radius * math.cos(angle)
            y = self.position[1] + radius * math.sin(angle)
            glVertex3f(x, y, self.position[2] + self.size[2] + 2)
        glEnd()
        glLineWidth(1)

# =====================================================
# DRONE MODELİ VE FİZİK
# =====================================================

@dataclass
class DronePhysics:
    """Drone fizik özellikleri"""
    max_speed: float = 15.0
    max_acceleration: float = 3.0
    max_climb_rate: float = 5.0
    max_turn_rate: float = 90.0
    battery_capacity: float = 100.0
    communication_range: float = 100.0
    detection_range: float = 80.0  # Nesne tespit menzili
    attack_speed: float = 25.0  # Saldırı hızı
    min_safe_distance: float = 15.0  # Minimum güvenli mesafe

class DroneStatus(Enum):
    """Drone durumları"""
    IDLE = "IDLE"
    TAKEOFF = "TAKEOFF"
    FLYING = "FLYING"
    SEARCHING = "SEARCHING"
    ENGAGING = "ENGAGING"  # Hedefe saldırıyor
    RETURNING = "RETURNING"
    LANDING = "LANDING"
    EMERGENCY = "EMERGENCY"

class Drone:
    """3D Drone simülasyon sınıfı"""
    
    def __init__(self, drone_id: int, start_position: np.ndarray, color: tuple):
        self.id = drone_id
        self.position = np.array(start_position, dtype=float)
        self.velocity = np.zeros(3)
        self.acceleration = np.zeros(3)
        self.orientation = np.array([0, 0, 0])
        self.color = color
        
        self.physics = DronePhysics()
        self.battery = 100.0
        
        self.waypoints = []
        self.current_waypoint_index = 0
        self.mission_complete = False
        self.path_history = deque(maxlen=100)
        
        self.nearby_drones = []
        self.collision_avoidance_vector = np.zeros(3)
        
        self.status = DroneStatus.IDLE
        self.home_position = np.copy(start_position)
        
        self.propeller_angle = 0
        
        # Hedef takip sistemi
        self.current_target = None
        self.detected_objects = []
        self.engagement_group = []  # Birlikte saldıran drone'lar
        self.is_group_leader = False
        self.attack_position = None
    
    def detect_ground_objects(self, ground_objects: List[GroundObject]) -> Optional[GroundObject]:
        """Yer nesnelerini tespit et"""
        self.detected_objects = []
        nearest_target = None
        nearest_distance = float('inf')
        
        for obj in ground_objects:
            if obj.destroyed:
                continue
                
            distance = np.linalg.norm(self.position[:2] - obj.position[:2])
            
            # Tespit menzilinde mi?
            if distance <= self.physics.detection_range:
                self.detected_objects.append(obj)
                
                # En yakın hedef (araba veya kamyon)
                if obj.type == ObjectType.CAR and not obj.is_engaged:
                    if distance < nearest_distance:
                        nearest_distance = distance
                        nearest_target = obj
        
        return nearest_target
    
    def coordinate_attack(self, target: GroundObject, all_drones: List['Drone']):
        """Koordineli saldırı planla"""
        if target.is_engaged:
            return
        
        # Yakındaki drone'ları bul
        nearby_attack_drones = []
        for drone in all_drones:
            if drone.id == self.id:
                continue
            
            distance = np.linalg.norm(self.position - drone.position)
            if distance < 50 and drone.status in [DroneStatus.FLYING, DroneStatus.SEARCHING]:
                # Hedefi görebiliyor mu?
                target_dist = np.linalg.norm(drone.position[:2] - target.position[:2])
                if target_dist < drone.physics.detection_range:
                    nearby_attack_drones.append(drone)
        
        # En yakın 2-3 drone'u seç
        nearby_attack_drones.sort(key=lambda d: np.linalg.norm(d.position - target.position))
        attack_group = nearby_attack_drones[:min(2, len(nearby_attack_drones))]
        attack_group.append(self)  # Kendini ekle
        
        # Saldırı pozisyonları belirle
        angle_step = 360 / len(attack_group)
        for i, drone in enumerate(attack_group):
            angle = math.radians(i * angle_step)
            offset = np.array([
                math.cos(angle) * 5,
                math.sin(angle) * 5,
                0
            ])
            drone.attack_position = target.position + offset
            drone.current_target = target
            drone.status = DroneStatus.ENGAGING
            drone.is_group_leader = (drone.id == self.id)
            
            # Hedefe drone'u ekle
            if drone.id not in target.engaged_by:
                target.engaged_by.append(drone.id)
        
        target.is_engaged = True
        
        print(f"[SALDIRI] Drone #{self.id} liderliğinde {len(attack_group)} drone, "
              f"{target.type.value} #{target.id} hedefine saldırıyor!")
    
    def execute_attack(self, dt: float):
        """Hedefe saldır"""
        if not self.current_target or self.current_target.destroyed:
            self.status = DroneStatus.FLYING
            self.current_target = None
            return
        
        # Hedefe hızlı yaklaş
        direction = self.current_target.position - self.position
        distance = np.linalg.norm(direction)
        
        if distance < 3:  # Çarpma mesafesi
            # Hasar ver
            damage = 50 * dt
            if self.current_target.take_damage(damage):
                print(f"[HEDEF YOK EDİLDİ] {self.current_target.type.value} #{self.current_target.id}")
            
            # Geri çekil
            self.velocity = -direction / distance * 10
            if distance < 1:
                self.status = DroneStatus.FLYING
                self.current_target = None
        else:
            # Hedefe doğru hızlan
            if distance > 0:
                desired_velocity = (direction / distance) * self.physics.attack_speed
                self.acceleration = (desired_velocity - self.velocity) * 5.0
                
                # İvmeyi sınırla
                acc_mag = np.linalg.norm(self.acceleration)
                if acc_mag > self.physics.max_acceleration * 2:
                    self.acceleration = (self.acceleration / acc_mag) * self.physics.max_acceleration * 2
    
    def update(self, dt: float, all_drones: List['Drone'], ground_objects: List[GroundObject]):
        """Drone güncelleme"""
        
        # Pervane animasyonu
        if self.status != DroneStatus.IDLE:
            self.propeller_angle += dt * 720
            self.battery -= dt * 0.5
        
        # Komşu drone'ları tespit et
        self.detect_nearby_drones(all_drones)
        
        # Çarpışma önleme
        if len(self.nearby_drones) > 0 and self.status != DroneStatus.ENGAGING:
            self.calculate_collision_avoidance()
        
        # Duruma göre davranış
        if self.status == DroneStatus.ENGAGING:
            self.execute_attack(dt)
        elif self.status in [DroneStatus.FLYING, DroneStatus.SEARCHING]:
            # Hedef ara
            target = self.detect_ground_objects(ground_objects)
            if target and target.type == ObjectType.CAR:
                self.coordinate_attack(target, all_drones)
            elif len(self.waypoints) > 0:
                self.navigate_to_waypoint(dt)
        
        # Fizik güncellemesi
        self.update_physics(dt)
        
        # Pozisyon geçmişi
        self.path_history.append(np.copy(self.position))
    
    def detect_nearby_drones(self, all_drones: List['Drone']):
        """Yakındaki drone'ları tespit et"""
        self.nearby_drones = []
        danger_radius = self.physics.min_safe_distance
        
        for other in all_drones:
            if other.id != self.id:
                distance = np.linalg.norm(self.position - other.position)
                if distance < danger_radius:
                    self.nearby_drones.append({
                        'drone': other,
                        'distance': distance
                    })
    
    def calculate_collision_avoidance(self):
        """Çarpışma önleme"""
        avoidance = np.zeros(3)
        
        for nearby in self.nearby_drones:
            other = nearby['drone']
            distance = nearby['distance']
            
            if distance < 8.0:
                direction = self.position - other.position
                if np.linalg.norm(direction) > 0:
                    direction = direction / np.linalg.norm(direction)
                    force = (8.0 - distance) / 8.0
                    avoidance += direction * force * 8.0
        
        self.collision_avoidance_vector = avoidance
    
    def navigate_to_waypoint(self, dt: float):
        """Waypoint navigasyonu"""
        if self.current_waypoint_index >= len(self.waypoints):
            self.mission_complete = True
            self.status = DroneStatus.SEARCHING
            return
        
        target = self.waypoints[self.current_waypoint_index]
        direction = target - self.position
        distance = np.linalg.norm(direction)
        
        if distance < 3.0:
            self.current_waypoint_index += 1
            return
        
        if distance > 0:
            desired_velocity = (direction / distance) * self.physics.max_speed
            desired_velocity += self.collision_avoidance_vector
            
            speed = np.linalg.norm(desired_velocity)
            if speed > self.physics.max_speed:
                desired_velocity = (desired_velocity / speed) * self.physics.max_speed
            
            self.acceleration = (desired_velocity - self.velocity) * 2.0
            
            acc_mag = np.linalg.norm(self.acceleration)
            if acc_mag > self.physics.max_acceleration:
                self.acceleration = (self.acceleration / acc_mag) * self.physics.max_acceleration
    
    def update_physics(self, dt: float):
        """Fizik güncellemesi"""
        self.velocity += self.acceleration * dt
        
        # Saldırı durumunda hız limiti farklı
        max_vel = self.physics.attack_speed if self.status == DroneStatus.ENGAGING else self.physics.max_speed
        speed = np.linalg.norm(self.velocity)
        if speed > max_vel:
            self.velocity = (self.velocity / speed) * max_vel
        
        self.position += self.velocity * dt
        
        if self.position[2] < 0:
            self.position[2] = 0
        elif self.position[2] > 150:
            self.position[2] = 150
        
        if speed > 0.1:
            self.orientation[2] = math.degrees(math.atan2(self.velocity[1], self.velocity[0]))
    
    def draw(self):
        """Drone çizimi"""
        glPushMatrix()
        glTranslatef(*self.position)
        
        glRotatef(self.orientation[2], 0, 0, 1)
        
        # Saldırı modundaysa kırmızı parlasın
        if self.status == DroneStatus.ENGAGING:
            pulse = abs(math.sin(time.time() * 10))
            glColor3f(1, pulse * 0.5, 0)
        else:
            glColor3f(*self.color)
        
        # Ana gövde
        size = 1.5
        self.draw_cube(size)
        
        # Pervaneler
        glColor3f(0.3, 0.3, 0.3)
        for i in range(4):
            angle = i * 90
            glPushMatrix()
            glRotatef(angle, 0, 0, 1)
            
            glBegin(GL_LINES)
            glVertex3f(0, 0, 0)
            glVertex3f(size * 3, 0, 0)
            glEnd()
            
            glTranslatef(size * 3, 0, 0)
            self.draw_cube(size * 0.4)
            
            glPushMatrix()
            glRotatef(self.propeller_angle, 0, 0, 1)
            glColor3f(0.7, 0.7, 0.7)
            glBegin(GL_QUADS)
            glVertex3f(-size * 1.5, -size * 0.1, size * 0.5)
            glVertex3f(size * 1.5, -size * 0.1, size * 0.5)
            glVertex3f(size * 1.5, size * 0.1, size * 0.5)
            glVertex3f(-size * 1.5, size * 0.1, size * 0.5)
            glVertex3f(-size * 0.1, -size * 1.5, size * 0.5)
            glVertex3f(size * 0.1, -size * 1.5, size * 0.5)
            glVertex3f(size * 0.1, size * 1.5, size * 0.5)
            glVertex3f(-size * 0.1, size * 1.5, size * 0.5)
            glEnd()
            glPopMatrix()
            
            glPopMatrix()
        
        glPopMatrix()
        
        # Rota çizgisi
        if len(self.path_history) > 1:
            glLineWidth(2)
            glBegin(GL_LINE_STRIP)
            for i, pos in enumerate(self.path_history):
                alpha = i / len(self.path_history)
                glColor4f(self.color[0], self.color[1], self.color[2], alpha * 0.5)
                glVertex3f(*pos)
            glEnd()
            glLineWidth(1)
        
        # Tespit alanı (sadece arama modunda)
        if self.status == DroneStatus.SEARCHING:
            self.draw_detection_range()
    
    def draw_detection_range(self):
        """Tespit alanını çiz"""
        glColor4f(0, 1, 0, 0.1)
        glBegin(GL_LINE_LOOP)
        for i in range(32):
            angle = i * math.pi * 2 / 32
            x = self.position[0] + self.physics.detection_range * math.cos(angle)
            y = self.position[1] + self.physics.detection_range * math.sin(angle)
            glVertex3f(x, y, 0)
        glEnd()
    
    def draw_cube(self, size):
        """Küp çizimi"""
        glBegin(GL_QUADS)
        # Ön
        glVertex3f(-size, -size, size)
        glVertex3f(size, -size, size)
        glVertex3f(size, size, size)
        glVertex3f(-size, size, size)
        # Arka
        glVertex3f(-size, -size, -size)
        glVertex3f(size, -size, -size)
        glVertex3f(size, size, -size)
        glVertex3f(-size, size, -size)
        # Sol
        glVertex3f(-size, -size, -size)
        glVertex3f(-size, -size, size)
        glVertex3f(-size, size, size)
        glVertex3f(-size, size, -size)
        # Sağ
        glVertex3f(size, -size, -size)
        glVertex3f(size, -size, size)
        glVertex3f(size, size, size)
        glVertex3f(size, size, -size)
        # Üst
        glVertex3f(-size, size, -size)
        glVertex3f(-size, size, size)
        glVertex3f(size, size, size)
        glVertex3f(size, size, -size)
        # Alt
        glVertex3f(-size, -size, -size)
        glVertex3f(-size, -size, size)
        glVertex3f(size, -size, size)
        glVertex3f(size, -size, -size)
        glEnd()

# =====================================================
# GÖREV PLANLAMA
# =====================================================

class MissionPlanner:
    """Görev planlama"""
    
    def __init__(self, area_corners: List[Tuple[float, float]], altitude: float = 50):
        self.area_corners = area_corners
        self.altitude = altitude
        self.drone_sectors = {}
    
    def divide_area_for_drones(self, num_drones: int) -> Dict[int, List[np.ndarray]]:
        """Alan bölme ve rota oluşturma"""
        x_coords = [c[0] for c in self.area_corners]
        y_coords = [c[1] for c in self.area_corners]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        grid_cols = int(math.sqrt(num_drones * (max_x - min_x) / (max_y - min_y)))
        if grid_cols == 0:
            grid_cols = 1
        grid_rows = int(math.ceil(num_drones / grid_cols))
        
        cell_width = (max_x - min_x) / grid_cols
        cell_height = (max_y - min_y) / grid_rows
        
        drone_waypoints = {}
        drone_id = 0
        
        for row in range(grid_rows):
            for col in range(grid_cols):
                if drone_id >= num_drones:
                    break
                
                cell_min_x = min_x + col * cell_width
                cell_max_x = min_x + (col + 1) * cell_width
                cell_min_y = min_y + row * cell_height
                cell_max_y = min_y + (row + 1) * cell_height
                
                waypoints = self.create_zigzag_pattern(
                    cell_min_x, cell_max_x,
                    cell_min_y, cell_max_y,
                    self.altitude,
                    spacing=15
                )
                
                drone_waypoints[drone_id] = waypoints
                drone_id += 1
        
        return drone_waypoints
    
    def create_zigzag_pattern(self, min_x, max_x, min_y, max_y, altitude, spacing):
        """Zigzag desen"""
        waypoints = []
        waypoints.append(np.array([(min_x + max_x) / 2, (min_y + max_y) / 2, altitude]))
        
        num_lines = int((max_y - min_y) / spacing)
        if num_lines == 0:
            num_lines = 1
        
        for i in range(num_lines + 1):
            y = min_y + i * spacing
            if y > max_y:
                y = max_y
            
            if i % 2 == 0:
                waypoints.append(np.array([min_x, y, altitude]))
                waypoints.append(np.array([max_x, y, altitude]))
            else:
                waypoints.append(np.array([max_x, y, altitude]))
                waypoints.append(np.array([min_x, y, altitude]))
        
        return waypoints

# =====================================================
# ANA SİMÜLASYON
# =====================================================

class DroneSwarmSimulator:
    """Ana Simülasyon Sınıfı"""
    
    def __init__(self, num_drones: int = 30):
        self.num_drones = num_drones
        self.drones = []
        self.ground_objects = []
        self.mission_planner = None
        self.simulation_time = 0
        self.paused = False
        
        # Nesne spawn ayarları
        self.last_spawn_time = 0
        self.spawn_interval = 5.0  # 5 saniyede bir yeni nesne grubu
        self.object_id_counter = 0
        
        # İstatistikler
        self.cars_destroyed = 0
        self.total_cars_spawned = 0
        
        pygame.init()
        self.display = (1400, 800)
        self.screen = pygame.display.set_mode(self.display, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("30 Drone Swarm - Hedef İmha Simülasyonu")
        
        self.camera_distance = 200
        self.camera_rotation_x = 30
        self.camera_rotation_z = 45
        self.camera_target = [150, 150, 25]
        
        self.setup_opengl()
        
        self.show_info = True
        self.show_grid = True
        self.show_paths = True
        self.selected_drone = None
        
        self.clock = pygame.time.Clock()
        self.fps = 60
    
    def setup_opengl(self):
        """OpenGL ayarları"""
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glClearColor(0.1, 0.1, 0.2, 1.0)
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
    
    def setup_camera(self):
        """Kamera ayarları"""
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60, (self.display[0] / self.display[1]), 0.1, 1000.0)
        
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        rad_x = math.radians(self.camera_rotation_x)
        rad_z = math.radians(self.camera_rotation_z)
        
        camera_x = self.camera_target[0] + self.camera_distance * math.cos(rad_x) * math.cos(rad_z)
        camera_y = self.camera_target[1] + self.camera_distance * math.cos(rad_x) * math.sin(rad_z)
        camera_z = self.camera_target[2] + self.camera_distance * math.sin(rad_x)
        
        gluLookAt(camera_x, camera_y, camera_z,
                  self.camera_target[0], self.camera_target[1], self.camera_target[2],
                  0, 0, 1)
    
    def initialize_drones(self):
        """Drone'ları başlat"""
        self.drones = []
        
        colors = []
        for i in range(self.num_drones):
            hue = i / self.num_drones
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            colors.append(rgb)
        
        grid_size = int(math.ceil(math.sqrt(self.num_drones)))
        spacing = 15
        
        for i in range(self.num_drones):
            row = i // grid_size
            col = i % grid_size
            
            start_pos = np.array([
                col * spacing,
                row * spacing,
                0
            ])
            
            drone = Drone(i, start_pos, colors[i])
            self.drones.append(drone)
        
        print(f"✓ {self.num_drones} drone başlatıldı")
    
    def spawn_ground_objects(self):
        """Yer nesnelerini oluştur"""
        current_time = self.simulation_time
        
        # Spawn zamanı geldi mi?
        if current_time - self.last_spawn_time < self.spawn_interval:
            return
        
        self.last_spawn_time = current_time
        
        # Ölü nesneleri temizle
        self.ground_objects = [obj for obj in self.ground_objects if not (obj.destroyed and obj.is_target)]
        
        # Yeni nesne grubu oluştur
        num_objects = random.randint(3, 7)
        
        for _ in range(num_objects):
            # Nesne tipi (ağırlıklı olarak araba)
            rand = random.random()
            if rand < 0.4:  # %40 araba
                obj_type = ObjectType.CAR
                self.total_cars_spawned += 1
            elif rand < 0.6:  # %20 kamyon
                obj_type = ObjectType.TRUCK
            elif rand < 0.75:  # %15 bina
                obj_type = ObjectType.BUILDING
            elif rand < 0.9:  # %15 ağaç
                obj_type = ObjectType.TREE
            else:  # %10 sivil
                obj_type = ObjectType.CIVILIAN
            
            # Rastgele pozisyon
            position = np.array([
                random.uniform(50, 250),
                random.uniform(50, 250),
                0
            ])
            
            # Nesneyi oluştur
            obj = GroundObject(self.object_id_counter, obj_type, position)
            self.object_id_counter += 1
            self.ground_objects.append(obj)
        
        print(f"[SPAWN] {num_objects} yeni nesne oluşturuldu "
              f"(Toplam araba: {sum(1 for o in self.ground_objects if o.type == ObjectType.CAR)})")
    
    def setup_mission(self):
        """Görev ayarları"""
        area_corners = [
            (50, 50),
            (250, 50),
            (250, 250),
            (50, 250)
        ]
        
        self.mission_planner = MissionPlanner(area_corners, altitude=40)
        drone_routes = self.mission_planner.divide_area_for_drones(self.num_drones)
        
        for drone_id, waypoints in drone_routes.items():
            if drone_id < len(self.drones):
                self.drones[drone_id].waypoints = waypoints
                self.drones[drone_id].status = DroneStatus.FLYING
        
        # İlk nesne grubunu oluştur
        self.spawn_ground_objects()
        
        print(f"✓ Görev planlaması tamamlandı")
    
    def update_simulation(self, dt):
        """Simülasyon güncelleme"""
        if not self.paused:
            self.simulation_time += dt
            
            # Nesne spawn
            self.spawn_ground_objects()
            
            # Nesneleri güncelle
            for obj in self.ground_objects:
                obj.update(dt, (50, 250, 50, 250))
                
                # Yok edilen arabaları say
                if obj.destroyed and obj.type == ObjectType.CAR and obj.id not in [self.cars_destroyed]:
                    self.cars_destroyed += 1
            
            # Drone'ları güncelle
            for drone in self.drones:
                drone.update(dt, self.drones, self.ground_objects)
            
            self.update_statistics()
    
    def update_statistics(self):
        """İstatistikleri güncelle"""
        active_drones = sum(1 for d in self.drones if d.status == DroneStatus.FLYING)
        engaging_drones = sum(1 for d in self.drones if d.status == DroneStatus.ENGAGING)
        active_cars = sum(1 for o in self.ground_objects if o.type == ObjectType.CAR and not o.destroyed)
        
        if int(self.simulation_time) % 5 == 0 and int(self.simulation_time * 10) % 10 == 0:
            print(f"[T={self.simulation_time:.1f}s] "
                  f"Devriye: {active_drones}, Saldırı: {engaging_drones}, "
                  f"Aktif Araba: {active_cars}, İmha: {self.cars_destroyed}/{self.total_cars_spawned}")
    
    def draw_scene(self):
        """Sahneyi çiz"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        self.setup_camera()
        
        if self.show_grid:
            self.draw_grid()
        
        if self.mission_planner:
            self.draw_mission_area()
        
        # Yer nesnelerini çiz
        for obj in self.ground_objects:
            obj.draw()
        
        # Drone'ları çiz
        for drone in self.drones:
            drone.draw()
        
        if self.show_info:
            self.draw_ui_info()
    
    def draw_grid(self):
        """Grid çizimi"""
        glLineWidth(1)
        glColor4f(0.3, 0.3, 0.3, 0.5)
        glBegin(GL_LINES)
        
        grid_size = 300
        grid_step = 20
        
        for i in range(-grid_size, grid_size + 1, grid_step):
            glVertex3f(i, -grid_size, 0)
            glVertex3f(i, grid_size, 0)
            glVertex3f(-grid_size, i, 0)
            glVertex3f(grid_size, i, 0)
        
        glEnd()
        
        glLineWidth(3)
        glBegin(GL_LINES)
        glColor3f(1, 0, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(50, 0, 0)
        glColor3f(0, 1, 0)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 50, 0)
        glColor3f(0, 0, 1)
        glVertex3f(0, 0, 0)
        glVertex3f(0, 0, 50)
        glEnd()
        glLineWidth(1)
    
    def draw_mission_area(self):
        """Görev alanı çizimi"""
        corners = self.mission_planner.area_corners
        altitude = self.mission_planner.altitude
        
        glLineWidth(2)
        
        glColor4f(1, 1, 0, 0.8)
        glBegin(GL_LINE_LOOP)
        for corner in corners:
            glVertex3f(corner[0], corner[1], 0)
        glEnd()
        
        glColor4f(0, 1, 1, 0.8)
        glBegin(GL_LINE_LOOP)
        for corner in corners:
            glVertex3f(corner[0], corner[1], altitude)
        glEnd()
        
        glColor4f(0.5, 0.5, 1, 0.5)
        glBegin(GL_LINES)
        for corner in corners:
            glVertex3f(corner[0], corner[1], 0)
            glVertex3f(corner[0], corner[1], altitude)
        glEnd()
        
        glLineWidth(1)
    
    def draw_ui_info(self):
        """UI bilgileri"""
        pass
    
    def handle_input(self):
        """Kullanıcı girişleri"""
        keys = pygame.key.get_pressed()
        
        if keys[K_LEFT]:
            self.camera_rotation_z -= 2
        if keys[K_RIGHT]:
            self.camera_rotation_z += 2
        if keys[K_UP]:
            self.camera_rotation_x = min(89, self.camera_rotation_x + 2)
        if keys[K_DOWN]:
            self.camera_rotation_x = max(-89, self.camera_rotation_x - 2)
        
        if keys[K_EQUALS] or keys[K_PLUS]:
            self.camera_distance = max(10, self.camera_distance - 5)
        if keys[K_MINUS]:
            self.camera_distance = min(500, self.camera_distance + 5)
        
        move_speed = 5
        if keys[K_w]:
            self.camera_target[1] += move_speed
        if keys[K_s]:
            self.camera_target[1] -= move_speed
        if keys[K_a]:
            self.camera_target[0] -= move_speed
        if keys[K_d]:
            self.camera_target[0] += move_speed
        if keys[K_q]:
            self.camera_target[2] -= move_speed
        if keys[K_e]:
            self.camera_target[2] += move_speed
    
    def run(self):
        """Ana döngü"""
        print("=" * 60)
        print("30 DRONE SWARM - HEDEF İMHA SİMÜLASYONU")
        print("=" * 60)
        print("\n[GÖREV]")
        print("- Drone'lar alanı tarar ve ARABA hedeflerini tespit eder")
        print("- İlk tespit eden drone, yakındaki 2-3 drone ile koordineli saldırır")
        print("- Diğer nesneler (bina, ağaç, sivil) engel olarak görev yapar")
        print("\n[KONTROLLER]")
        print("- Ok tuşları: Kamera döndürme")
        print("- W/A/S/D: Kamera hareketi")
        print("- Q/E: Yükseklik")
        print("- +/-: Zoom")
        print("- SPACE: Başlat/Durdur")
        print("- G: Grid aç/kapa")
        print("- R: Simülasyonu sıfırla")
        print("- ESC: Çıkış")
        print("=" * 60)
        
        self.initialize_drones()
        self.setup_mission()
        
        running = True
        dt = 0
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                        status = "DURDURULDU" if self.paused else "DEVAM"
                        print(f"Simülasyon {status}")
                    elif event.key == pygame.K_g:
                        self.show_grid = not self.show_grid
                    elif event.key == pygame.K_r:
                        print("Simülasyon sıfırlanıyor...")
                        self.initialize_drones()
                        self.setup_mission()
                        self.simulation_time = 0
                        self.cars_destroyed = 0
                        self.total_cars_spawned = 0
            
            self.handle_input()
            self.update_simulation(dt)
            self.draw_scene()
            
            pygame.display.flip()
            dt = self.clock.tick(self.fps) / 1000.0
        
        pygame.quit()
        print(f"\n[SONUÇ] Toplam {self.cars_destroyed} araba imha edildi!")
        print("Simülasyon sonlandı.")

# =====================================================
# ÇALIŞTIR
# =====================================================

if __name__ == "__main__":
    simulator = DroneSwarmSimulator(num_drones=30)
    simulator.run()
