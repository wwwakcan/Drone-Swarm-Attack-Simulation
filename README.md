# Drone Swarm Attack Simulation 🚁

Drone sürülerinin koordineli hareket ve saldırı simülasyonu projesi. Sürü formasyonu, hedefe grup saldırısı ve engelden kaçınma özelliklerini içerir.

## 🚀 Kurulum

### Gerekli Paketleri Yükleyin

```bash
pip3 install flask flask-socketio flask-cors pygame PyOpenGL PyOpenGL-accelerate numpy python-socketio eventlet
```

## 💻 Kullanım

Proje iki farklı simülasyon modu sunar:

### 1. 3D Native Simülasyon

3D görselleştirme ile masaüstü simülasyon:

```bash
python3 example_native.py
```

Bu komut direkt olarak 3D simülasyon penceresini açar.

### 2. Web Tabanlı Simülasyon

Tarayıcı üzerinden çalışan web simülasyonu:

```bash
python3 example_web.py
```

Simülasyon başladıktan sonra tarayıcınızda şu adrese gidin:

```
http://localhost:1010
```

## 📋 Özellikler

- **Sürü Formasyonu**: Drone'ların koordineli grup uçuşu
- **Koordineli Saldırı**: Hedefe senkronize saldırı
- **Grup Saldırısı**: Hedefe yakın drone'ların birlikte saldırısı
- **Engelden Kaçınma**: Otomatik engel algılama ve kaçınma

## 🖥️ Sistem Gereksinimleri

- Python 3.x
- OpenGL destekli grafik kartı (3D simülasyon için)
- Modern web tarayıcı (Web simülasyon için)

## 📝 Notlar

- Web simülasyonu `localhost:1010` portunda çalışır
- 3D simülasyon için OpenGL desteği gereklidir
- Her iki simülasyon da gerçek zamanlı olarak çalışır

## 📄 Lisans

MIT License

## 👤 Geliştirici

[@wwwakcan](https://github.com/wwwakcan)

## 🤖 AI Desteği

Bu proje Claude AI desteği ile geliştirilmiştir.
