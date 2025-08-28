"""
DRONE SWARM SİMÜLASYON - TAM ÇALIŞAN VERSİYON
============================================
30 Drone Koordineli Saldırı Simülasyonu

Kurulum:
pip install flask flask-socketio flask-cors numpy

Çalıştırma:
python drone_sim.py

Sonra tarayıcıda: http://localhost:1010
"""

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import numpy as np
import random
import time
import math
import threading
import json
from datetime import datetime
import colorsys

app = Flask(__name__)
app.config['SECRET_KEY'] = 'drone-sim-2024'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
CORS(app)

# =====================================================
# ARAÇ (HEDEF) SINIFI
# =====================================================

class VehicleTarget:
    """Hedef araç sınıfı"""
    def __init__(self, vehicle_id, position=None, vehicle_type="truck"):
        self.id = vehicle_id
        if position:
            self.position = np.array(position, dtype=float)
        else:
            self.position = np.array([
                random.uniform(100, 300),
                random.uniform(100, 300),
                0
            ], dtype=float)
        
        self.vehicle_type = vehicle_type
        self.velocity = np.array([
            random.uniform(-5, 5),
            random.uniform(-5, 5),
            0
        ], dtype=float)
        
        # Araç özellikleri
        if vehicle_type == "truck":
            self.size = (8, 3, 3)
            self.health = 400  # 4 drone gerekli
            self.max_health = 400
            self.armor = 50
        elif vehicle_type == "car":
            self.size = (4, 2, 1.5)
            self.health = 100
            self.max_health = 100
            self.armor = 20
        else:  # tank
            self.size = (10, 4, 3)
            self.health = 600
            self.max_health = 600
            self.armor = 100
        
        self.destroyed = False
        self.engaged_by = []
        self.spawn_time = time.time()
    
    def update(self, dt):
        """Aracı güncelle"""
        if not self.destroyed:
            # Pozisyon güncelleme
            self.position += self.velocity * dt
            
            # Sınır kontrolü ve yansıma
            if self.position[0] < 50 or self.position[0] > 350:
                self.velocity[0] *= -1
                self.position[0] = np.clip(self.position[0], 50, 350)
            
            if self.position[1] < 50 or self.position[1] > 350:
                self.velocity[1] *= -1
                self.position[1] = np.clip(self.position[1], 50, 350)
    
    def take_damage(self, damage):
        """Hasar al"""
        # Zırh etkisi
        actual_damage = damage * (1 - self.armor / 200)
        self.health -= actual_damage
        
        if self.health <= 0:
            self.health = 0
            self.destroyed = True
            return True
        return False
    
    def to_dict(self):
        """JSON için dict'e çevir"""
        return {
            'id': self.id,
            'position': self.position.tolist(),
            'velocity': self.velocity.tolist(),
            'health': float(self.health),
            'max_health': float(self.max_health),
            'destroyed': bool(self.destroyed),
            'type': self.vehicle_type,
            'engaged_by': self.engaged_by
        }

# =====================================================
# DRONE SINIFI
# =====================================================

class Drone:
    """Gelişmiş drone sınıfı"""
    def __init__(self, drone_id):
        self.id = drone_id
        
        # Başlangıç pozisyonu (grid formasyonu)
        row = drone_id // 6
        col = drone_id % 6
        self.position = np.array([
            col * 20 + 50,
            row * 20 + 50,
            0
        ], dtype=float)
        
        self.velocity = np.zeros(3)
        self.battery = 100.0
        self.status = "IDLE"
        self.target = None
        self.armed = False
        self.color = colorsys.hsv_to_rgb(drone_id / 30, 0.8, 0.9)
        
        # Fizik parametreleri
        self.max_speed = 15.0
        self.attack_speed = 30.0
        self.detection_range = 100.0
        
        # Patlayıcı
        self.explosive_weight = 1.0  # kg
        self.explosive_damage = 100.0
        self.detonated = False
        
        # Devriye noktaları
        self.waypoints = [
            [self.position[0], self.position[1], 50],
            [self.position[0] + 100, self.position[1], 50],
            [self.position[0] + 100, self.position[1] + 100, 50],
            [self.position[0], self.position[1] + 100, 50]
        ]
        self.current_waypoint = 0
        self.home_position = np.copy(self.position)
    
    def detect_and_engage(self, vehicles, all_drones):
        """Hedef tespit et ve koordineli saldırı planla"""
        if self.target:
            return
        
        for vehicle in vehicles:
            if vehicle.destroyed:
                continue
            
            # Mesafe kontrolü
            distance = np.linalg.norm(self.position[:2] - vehicle.position[:2])
            
            if distance < self.detection_range and len(vehicle.engaged_by) < 4:
                # Hedefi seç
                self.target = vehicle
                self.status = "ENGAGING"
                self.armed = True
                vehicle.engaged_by.append(self.id)
                
                # Yakındaki drone'ları çağır (3-4 drone koordineli)
                recruited = 1
                for other_drone in all_drones:
                    if recruited >= 3:  # Max 4 drone
                        break
                    
                    if (other_drone.id != self.id and 
                        other_drone.status in ["FLYING", "SEARCHING"] and
                        not other_drone.target):
                        
                        other_dist = np.linalg.norm(
                            other_drone.position - vehicle.position
                        )
                        
                        if other_dist < 150:  # 150m menzil
                            other_drone.target = vehicle
                            other_drone.status = "ENGAGING"
                            other_drone.armed = True
                            vehicle.engaged_by.append(other_drone.id)
                            recruited += 1
                
                print(f"[SALDIRI] Drone #{self.id} liderliğinde {recruited+1} drone, "
                      f"{vehicle.vehicle_type} #{vehicle.id} hedefine koordineli saldırı!")
                break
    
    def attack_target(self, dt):
        """Hedefe saldır"""
        if not self.target:
            return
        
        if self.target.destroyed:
            self.status = "RETURNING"
            self.target = None
            self.armed = False
            return
        
        # Farklı açılardan yaklaşma
        angle_offset = (self.id % 4) * 90
        angle_rad = math.radians(angle_offset)
        
        target_pos = self.target.position + np.array([
            math.cos(angle_rad) * 5,
            math.sin(angle_rad) * 5,
            2
        ])
        
        direction = target_pos - self.position
        distance = np.linalg.norm(direction)
        
        if distance < 3:  # Çarpma mesafesi
            if not self.detonated:
                # Patlama!
                self.detonated = True
                if self.target.take_damage(self.explosive_damage):
                    print(f"[İMHA] {self.target.vehicle_type} #{self.target.id} yok edildi!")
                self.status = "DESTROYED"
                self.velocity = np.zeros(3)
        else:
            # Hedefe hızlı yaklaş
            direction_norm = direction / (distance + 0.001)
            velocity_magnitude = self.attack_speed
            self.velocity = direction_norm * velocity_magnitude
    
    def patrol(self, dt):
        """Devriye yap"""
        if self.current_waypoint >= len(self.waypoints):
            self.current_waypoint = 0
        
        target = np.array(self.waypoints[self.current_waypoint])
        direction = target - self.position
        distance = np.linalg.norm(direction)
        
        if distance < 10:
            self.current_waypoint = (self.current_waypoint + 1) % len(self.waypoints)
        else:
            direction_norm = direction / (distance + 0.001)
            self.velocity = direction_norm * self.max_speed
    
    def return_home(self, dt):
        """Eve dön"""
        direction = self.home_position - self.position
        distance = np.linalg.norm(direction)
        
        if distance < 5:
            self.status = "IDLE"
            self.velocity = np.zeros(3)
            self.battery = 100.0  # Batarya doldur
        else:
            direction_norm = direction / (distance + 0.001)
            self.velocity = direction_norm * 20
    
    def update(self, dt, vehicles, all_drones):
        """Drone'u güncelle"""
        # Yok edildiyse güncelleme yapma
        if self.status == "DESTROYED":
            return
        
        # Batarya tüketimi
        if self.status not in ["IDLE", "DESTROYED"]:
            self.battery -= dt * 1.5
            if self.battery <= 10:
                self.status = "RETURNING"
                self.target = None
        
        # Durum makinesi
        if self.status == "IDLE":
            # Kalkış
            if self.position[2] < 50:
                self.position[2] += 10 * dt
            else:
                self.status = "FLYING"
        
        elif self.status == "FLYING" or self.status == "SEARCHING":
            # Hedef ara
            self.detect_and_engage(vehicles, all_drones)
            
            # Hedef yoksa devriye
            if not self.target:
                self.patrol(dt)
        
        elif self.status == "ENGAGING":
            # Saldır
            self.attack_target(dt)
        
        elif self.status == "RETURNING":
            # Eve dön
            self.return_home(dt)
        
        # Fizik güncelleme
        if self.status != "DESTROYED":
            self.position += self.velocity * dt
            
            # Pozisyon sınırları
            self.position[0] = np.clip(self.position[0], 0, 400)
            self.position[1] = np.clip(self.position[1], 0, 400)
            self.position[2] = np.clip(self.position[2], 0, 150)
    
    def to_dict(self):
        """JSON için dict'e çevir"""
        return {
            'id': self.id,
            'position': self.position.tolist(),
            'velocity': float(np.linalg.norm(self.velocity)),
            'battery': float(self.battery),
            'status': self.status,
            'target': self.target.id if self.target else None,
            'armed': bool(self.armed)
        }

# =====================================================
# ANA SİMÜLASYON SINIFI
# =====================================================

class DroneSimulation:
    """Ana simülasyon sınıfı"""
    def __init__(self, num_drones=30):
        self.num_drones = num_drones
        self.drones = []
        self.vehicles = []
        self.simulation_time = 0
        self.running = False
        self.paused = False
        self.auto_spawn = True
        self.last_spawn_time = 0
        self.spawn_interval = 10.0
        self.targets_destroyed = 0
        self.vehicle_id_counter = 0
        
        print(f"Simülasyon başlatılıyor: {num_drones} drone")
        self.initialize()
    
    def initialize(self):
        """Simülasyonu başlat"""
        # Drone'ları oluştur
        self.drones = []
        for i in range(self.num_drones):
            drone = Drone(i)
            self.drones.append(drone)
        
        # İlk hedefleri oluştur
        for i in range(5):
            self.spawn_vehicle()
        
        self.running = True
        print(f"✓ {len(self.drones)} drone hazır!")
        print(f"✓ {len(self.vehicles)} hedef oluşturuldu!")
    
    def spawn_vehicle(self, position=None, vehicle_type=None):
        """Yeni araç oluştur"""
        if vehicle_type is None:
            vehicle_type = random.choice(['truck', 'truck', 'car'])  # Ağırlıklı kamyon
        
        vehicle = VehicleTarget(self.vehicle_id_counter, position, vehicle_type)
        self.vehicles.append(vehicle)
        self.vehicle_id_counter += 1
        return vehicle
    
    def update(self, dt):
        """Simülasyonu güncelle"""
        if self.paused:
            return
        
        self.simulation_time += dt
        
        # Otomatik araç üretimi
        if self.auto_spawn and (self.simulation_time - self.last_spawn_time) > self.spawn_interval:
            # Ölü araçları temizle
            self.vehicles = [v for v in self.vehicles if not v.destroyed or 
                           (time.time() - v.spawn_time) < 5]
            
            # Yeni araçlar ekle
            if len(self.vehicles) < 8:  # Max 8 araç
                num_new = random.randint(2, 4)
                for _ in range(num_new):
                    self.spawn_vehicle()
                
                self.last_spawn_time = self.simulation_time
                print(f"[SPAWN] {num_new} yeni hedef. Toplam: {len(self.vehicles)}")
        
        # Araçları güncelle
        for vehicle in self.vehicles:
            vehicle.update(dt)
        
        # Drone'ları güncelle
        for drone in self.drones:
            drone.update(dt, self.vehicles, self.drones)
        
        # İmha sayısını güncelle
        destroyed = sum(1 for v in self.vehicles if v.destroyed)
        if destroyed > self.targets_destroyed:
            self.targets_destroyed = destroyed
    
    def get_stats(self):
        """İstatistikleri al"""
        return {
            'active': sum(1 for d in self.drones if d.status not in ["IDLE", "DESTROYED"]),
            'engaging': sum(1 for d in self.drones if d.status == "ENGAGING"),
            'targets': sum(1 for v in self.vehicles if not v.destroyed),
            'destroyed': self.targets_destroyed
        }
    
    def reset(self):
        """Simülasyonu sıfırla"""
        self.simulation_time = 0
        self.targets_destroyed = 0
        self.vehicle_id_counter = 0
        self.vehicles = []
        self.drones = []
        self.initialize()
        print("Simülasyon sıfırlandı!")

# Global simülasyon instance
simulation = None

# =====================================================
# FLASK ROUTE'LAR
# =====================================================

@app.route('/')
def index():
    """Ana sayfa"""
    return render_template('index.html')

@app.route('/api/status')
def api_status():
    """API durumu"""
    if simulation:
        return jsonify({
            'running': simulation.running,
            'paused': simulation.paused,
            'time': simulation.simulation_time,
            'drones': len(simulation.drones),
            'vehicles': len(simulation.vehicles)
        })
    return jsonify({'error': 'Simulation not initialized'})

# =====================================================
# SOCKET.IO EVENT'LER
# =====================================================

@socketio.on('connect')
def handle_connect():
    """Client bağlandı"""
    print(f'Client bağlandı: {request.sid}')
    emit('message', 'Sunucuya hoş geldiniz!')
    
    # İlk veriyi gönder
    if simulation:
        emit('simulation_update', {
            'time': float(simulation.simulation_time),
            'drones': [d.to_dict() for d in simulation.drones],
            'targets': [v.to_dict() for v in simulation.vehicles],
            'stats': simulation.get_stats()
        })

@socketio.on('disconnect')
def handle_disconnect():
    """Client ayrıldı"""
    print(f'Client ayrıldı: {request.sid}')

@socketio.on('request_update')
def handle_update_request():
    """Güncelleme isteği"""
    if simulation:
        emit('simulation_update', {
            'time': float(simulation.simulation_time),
            'drones': [d.to_dict() for d in simulation.drones],
            'targets': [v.to_dict() for v in simulation.vehicles],
            'stats': simulation.get_stats()
        })

@socketio.on('control')
def handle_control(data):
    """Kontrol komutları"""
    action = data.get('action')
    
    if action == 'pause':
        simulation.paused = not simulation.paused
        emit('message', f'Simülasyon {"duraklatıldı" if simulation.paused else "devam ediyor"}')
    
    elif action == 'reset':
        simulation.reset()
        emit('message', 'Simülasyon sıfırlandı')
    
    elif action == 'spawn_target':
        vehicle_data = data.get('data', {})
        position = vehicle_data.get('position')
        vehicle_type = vehicle_data.get('type', 'truck')
        
        simulation.spawn_vehicle(position, vehicle_type)
        emit('message', f'Yeni {vehicle_type} hedefi eklendi')
    
    elif action == 'auto_mode':
        simulation.auto_spawn = data.get('value', True)
        emit('message', f'Otomatik mod: {simulation.auto_spawn}')

# =====================================================
# BROADCAST THREAD
# =====================================================

def broadcast_updates():
    """Periyodik güncelleme yayını"""
    while True:
        try:
            if simulation and simulation.running:
                # Veriyi hazırla
                update_data = {
                    'time': float(simulation.simulation_time),
                    'drones': [d.to_dict() for d in simulation.drones],
                    'targets': [v.to_dict() for v in simulation.vehicles],
                    'stats': simulation.get_stats()
                }
                
                # Tüm client'lara gönder
                socketio.emit('simulation_update', update_data, broadcast=True)
                
        except Exception as e:
            print(f"Broadcast hatası: {e}")
        
        time.sleep(0.1)  # 100ms

# =====================================================
# SİMÜLASYON THREAD
# =====================================================

def simulation_loop():
    """Ana simülasyon döngüsü"""
    dt = 0.05  # 50ms = 20 FPS
    
    while simulation and simulation.running:
        if not simulation.paused:
            simulation.update(dt)
        time.sleep(dt)

# =====================================================
# ANA BAŞLATMA
# =====================================================

def initialize_all():
    """Tüm sistemleri başlat"""
    global simulation
    
    # Simülasyonu oluştur
    simulation = DroneSimulation(30)
    
    # Thread'leri başlat
    broadcast_thread = threading.Thread(target=broadcast_updates)
    broadcast_thread.daemon = True
    broadcast_thread.start()
    
    sim_thread = threading.Thread(target=simulation_loop)
    sim_thread.daemon = True
    sim_thread.start()
    
    print("✓ Tüm sistemler hazır!")

if __name__ == '__main__':
    print("=" * 50)
    print("DRONE SWARM SİMÜLASYON SUNUCUSU")
    print("=" * 50)
    print("30 Drone Koordineli Saldırı Sistemi")
    print("Web arayüzü: http://localhost:1010")
    print("=" * 50)
    
    # Sistemleri başlat
    initialize_all()
    
    # Web sunucusunu başlat
    socketio.run(app, debug=False, port=1010, host='0.0.0.0')