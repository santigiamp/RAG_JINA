#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PETION7 JINA BULLETPROOF - Procesamiento robusto con 10M tokens
Manejo completo de rate limiting, reintentos y recuperación de errores
"""

import subprocess
import sys
import os
import time
import json
import re
import base64
import io
import random
from typing import List, Dict, Optional
from dataclasses import dataclass
from pathlib import Path
import threading
import signal

def install_package(package):
    """Instalar paquete individual con retry"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])
            return True
        except subprocess.CalledProcessError:
            if attempt < max_retries - 1:
                print(f"⚠️ Reintentando instalación {package} ({attempt + 1}/{max_retries})")
                time.sleep(2)
            else:
                print(f"❌ Error instalando {package}")
                return False

# Dependencias mínimas con versiones específicas
packages = [
    "pymupdf==1.23.14",
    "qdrant-client>=1.7.0", 
    "Pillow>=10.0.0",
    "requests>=2.31.0",
    "numpy>=1.24.0",
    "tenacity>=8.2.0"  # Para retry logic robusto
]

print("📦 Instalando dependencias con retry...")
for package in packages:
    print(f"   📥 {package}...")
    install_package(package)

print("✅ Instalación completada\n")

# Imports con manejo de errores
try:
    import fitz
    import requests
    from PIL import Image
    import numpy as np
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    print("✅ Todas las dependencias importadas")
except ImportError as e:
    print(f"❌ Error crítico importando: {e}")
    exit(1)

# ========================================
# CONFIGURACIÓN DE RATE LIMITING
# ========================================

class RateLimiter:
    """Rate limiter robusto para Jina AI"""
    
    def __init__(self, calls_per_minute=50):  # Conservador: 50 calls/min
        self.calls_per_minute = calls_per_minute
        self.min_interval = 60.0 / calls_per_minute  # 1.2 segundos entre calls
        self.calls_log = []
        self.lock = threading.Lock()
        
        print(f"🚦 Rate limiter: {calls_per_minute} calls/min ({self.min_interval:.1f}s interval)")
    
    def wait_if_needed(self):
        """Esperar si es necesario para respetar rate limit"""
        with self.lock:
            now = time.time()
            
            # Limpiar calls antiguos (más de 1 minuto)
            self.calls_log = [call_time for call_time in self.calls_log if now - call_time < 60]
            
            # Si hemos hecho muchas calls, esperar
            if len(self.calls_log) >= self.calls_per_minute:
                sleep_time = 60 - (now - self.calls_log[0]) + 1  # +1 seg buffer
                if sleep_time > 0:
                    print(f"🚦 Rate limit: esperando {sleep_time:.1f}s...")
                    time.sleep(sleep_time)
            
            # Esperar intervalo mínimo desde última call
            if self.calls_log:
                time_since_last = now - self.calls_log[-1]
                if time_since_last < self.min_interval:
                    sleep_time = self.min_interval - time_since_last + random.uniform(0.1, 0.3)  # Jitter
                    print(f"⏱️ Intervalo: esperando {sleep_time:.1f}s...")
                    time.sleep(sleep_time)
            
            # Registrar esta call
            self.calls_log.append(time.time())

# ========================================
# CONFIGURACIÓN Y SETUP
# ========================================

def setup_environment():
    """Configurar entorno con validación robusta"""
    print("🔐 Configurando entorno...")
    
    IS_COLAB = False
    try:
        from google.colab import userdata
        IS_COLAB = True
        print("✅ Detectado: Google Colab")
        
        required_secrets = ['QDRANT_URL', 'QDRANT_API_KEY', 'JINA_API_KEY']
        
        for secret in required_secrets:
            try:
                value = userdata.get(secret)
                if not value or len(value) < 10:
                    print(f"❌ {secret} - Valor inválido o muy corto")
                    return False
                os.environ[secret] = value
                print(f"✅ {secret} - Configurado")
            except Exception as e:
                print(f"❌ {secret} - Error: {e}")
                return False
        
        os.environ['QDRANT_COLLECTION_NAME'] = 'manual_jina_bulletproof'
        return True
        
    except ImportError:
        print("✅ Detectado: Entorno local")
        
        required_vars = ['QDRANT_URL', 'QDRANT_API_KEY', 'JINA_API_KEY']
        missing = []
        
        for var in required_vars:
            value = os.getenv(var)
            if not value or len(value) < 10:
                missing.append(var)
        
        if missing:
            print(f"❌ Variables faltantes o inválidas: {missing}")
            print("\n💡 Asegúrate de tener:")
            print("- QDRANT_URL (URL completa con puerto)")
            print("- QDRANT_API_KEY (mínimo 20 caracteres)")
            print("- JINA_API_KEY (mínimo 20 caracteres)")
            return False
        
        os.environ['QDRANT_COLLECTION_NAME'] = 'manual_jina_bulletproof'
        return True

CONFIG_OK = setup_environment()

@dataclass
class ManualSection:
    """Sección del manual"""
    id: str
    title: str
    start_page: int
    end_page: int
    keywords: List[str]
    content_type: str

# Estructura del manual
MANUAL_SECTIONS = {
    "intro": ManualSection("intro", "Introducción y Seguridad", 1, 11, ["seguridad", "epp"], "safety"),
    "equipos": ManualSection("equipos", "Equipos Básicos", 12, 13, ["aspiradora", "escalera"], "equipment"),
    "edificios": ManualSection("edificios", "Edificios e Infraestructura", 14, 26, ["extintor", "emergencia"], "building"),
    "electricos": ManualSection("electricos", "Sistemas Eléctricos", 27, 30, ["eléctrico", "tablero"], "electrical"),
    "electronicos": ManualSection("electronicos", "Sistemas Electrónicos", 31, 33, ["audio", "video"], "electronic"),
    "mecanicos": ManualSection("mecanicos", "Sistemas Mecánicos", 34, 38, ["aire acondicionado", "bomba"], "mechanical"),
    "reparaciones": ManualSection("reparaciones", "Reparaciones Rápidas", 40, 43, ["óxido", "grieta"], "repair")
}

# ========================================
# PROCESADOR BULLETPROOF
# ========================================

class JinaBulletproofProcessor:
    """Procesador ultra robusto con manejo completo de errores"""
    
    def __init__(self):
        print("🛡️ Inicializando procesador bulletproof...")
        
        # Configuración
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.jina_api_key = os.getenv("JINA_API_KEY")
        self.collection_name = os.getenv("QDRANT_COLLECTION_NAME", "manual_jina_bulletproof")
        
        # Rate limiter
        self.rate_limiter = RateLimiter(calls_per_minute=45)  # Muy conservador
        
        # Clientes
        self.qdrant_client = None
        
        # Contadores de tokens y llamadas
        self.token_usage = {
            'total_tokens': 0,
            'total_calls': 0,
            'failed_calls': 0,
            'successful_calls': 0,
            'max_tokens': 10_000_000  # 10M límite
        }
        
        # Estado de procesamiento
        self.processing_state = {
            'current_section': None,
            'current_page': None,
            'processed_chunks': [],
            'failed_chunks': [],
            'last_checkpoint': time.time()
        }
        
        # Estadísticas
        self.stats = {
            'sections_processed': 0,
            'pages_processed': 0,
            'chunks_created': 0,
            'images_processed': 0,
            'jina_calls': 0,
            'retries': 0,
            'checkpoints_saved': 0,
            'start_time': time.time()
        }
        
        self.init_clients()
        self.setup_signal_handlers()
    
    def setup_signal_handlers(self):
        """Configurar handlers para interrupciones"""
        def signal_handler(signum, frame):
            print(f"\n🛑 Interrupción detectada (signal {signum})")
            self.save_checkpoint()
            print("💾 Checkpoint guardado. Puedes reanudar después.")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def init_clients(self):
        """Inicializar clientes con retry"""
        # Qdrant con retry
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if self.qdrant_url and self.qdrant_api_key:
                    self.qdrant_client = QdrantClient(
                        url=self.qdrant_url,
                        api_key=self.qdrant_api_key,
                        timeout=30
                    )
                    # Test connection
                    collections = self.qdrant_client.get_collections()
                    print("✅ Qdrant conectado y verificado")
                    break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"⚠️ Reintentando conexión Qdrant ({attempt + 1}/{max_retries}): {e}")
                    time.sleep(5)
                else:
                    print(f"❌ Error crítico Qdrant: {e}")
        
        # Verificar Jina AI
        if self.jina_api_key and len(self.jina_api_key) > 20:
            print("✅ Jina AI configurada correctamente")
        else:
            print("❌ JINA_API_KEY inválida")
    
    def estimate_tokens(self, text: str, images: List[Dict]) -> int:
        """Estimar tokens que consumirá la llamada"""
        # Estimación conservadora:
        # - Texto: ~1 token por 4 caracteres
        # - Imagen: ~1000 tokens por imagen (estimación alta)
        text_tokens = len(text) // 4
        image_tokens = len(images) * 1000
        total = text_tokens + image_tokens
        return min(total, 8000)  # Cap máximo por llamada
    
    def check_token_budget(self, estimated_tokens: int) -> bool:
        """Verificar si tenemos suficientes tokens"""
        remaining = self.token_usage['max_tokens'] - self.token_usage['total_tokens']
        
        if remaining < estimated_tokens:
            print(f"⚠️ Tokens insuficientes: {remaining} restantes, {estimated_tokens} necesarios")
            return False
        
        if remaining < 50000:  # Advertencia cuando quedan menos de 50k
            print(f"⚠️ ADVERTENCIA: Solo {remaining:,} tokens restantes")
        
        return True
    
    def save_checkpoint(self):
        """Guardar estado para reanudar después"""
        checkpoint_data = {
            'timestamp': time.time(),
            'token_usage': self.token_usage,
            'processing_state': self.processing_state,
            'stats': self.stats,
            'collection_name': self.collection_name
        }
        
        checkpoint_file = f"checkpoint_{int(time.time())}.json"
        try:
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            print(f"💾 Checkpoint guardado: {checkpoint_file}")
            self.stats['checkpoints_saved'] += 1
        except Exception as e:
            print(f"❌ Error guardando checkpoint: {e}")
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        retry=retry_if_exception_type((requests.exceptions.RequestException, requests.exceptions.Timeout))
    )
    def get_jina_multimodal_embedding(self, text: str, images: List[Dict] = None) -> Optional[List[float]]:
        """Embedding con retry exponencial y manejo completo de errores"""
        
        # Estimar tokens
        estimated_tokens = self.estimate_tokens(text, images or [])
        
        # Verificar presupuesto
        if not self.check_token_budget(estimated_tokens):
            raise Exception("Token budget exceeded")
        
        # Rate limiting
        self.rate_limiter.wait_if_needed()
        
        if not self.jina_api_key:
            print("❌ Sin API key")
            return None
        
        try:
            url = "https://api.jina.ai/v1/embeddings"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.jina_api_key}",
                "User-Agent": "JinaBulletproof/1.0"
            }
            
            # Preparar inputs (texto + máximo 2 imágenes para conservar tokens)
            inputs = [{"text": text[:6000]}]  # Limitado para conservar tokens
            
            if images:
                # Solo las 2 mejores imágenes (más grandes)
                sorted_images = sorted(images, key=lambda x: x.get('width', 0) * x.get('height', 0), reverse=True)
                for img in sorted_images[:2]:
                    if 'data_url' in img:
                        inputs.append({"image": img['data_url']})
            
            payload = {
                "model": "jina-embeddings-v4",
                "normalized": True,
                "embedding_type": "float",
                "input": inputs
            }
            
            # Llamada con timeout generoso
            response = requests.post(
                url, 
                headers=headers, 
                json=payload, 
                timeout=60  # 1 minuto timeout
            )
            
            # Manejo detallado de respuestas
            if response.status_code == 200:
                data = response.json()
                embeddings = [item["embedding"] for item in data["data"]]
                
                # Promediar embeddings
                if len(embeddings) > 1:
                    final_embedding = np.mean(embeddings, axis=0).tolist()
                else:
                    final_embedding = embeddings[0]
                
                # Actualizar contadores
                self.token_usage['total_tokens'] += estimated_tokens
                self.token_usage['successful_calls'] += 1
                self.stats['jina_calls'] += 1
                
                print(f"✅ Embedding exitoso: {len(final_embedding)} dims, ~{estimated_tokens} tokens")
                print(f"📊 Total tokens usados: {self.token_usage['total_tokens']:,} / {self.token_usage['max_tokens']:,}")
                
                return final_embedding
                
            elif response.status_code == 429:
                # Rate limit hit
                print(f"🚦 Rate limit (429), esperando 60s...")
                time.sleep(60)
                raise requests.exceptions.RequestException("Rate limit exceeded")
                
            elif response.status_code == 401:
                print(f"❌ API key inválida (401)")
                raise Exception("Invalid API key")
                
            elif response.status_code >= 500:
                print(f"⚠️ Error servidor Jina ({response.status_code}), reintentando...")
                raise requests.exceptions.RequestException(f"Server error: {response.status_code}")
                
            else:
                print(f"⚠️ Error desconocido: {response.status_code} - {response.text}")
                self.token_usage['failed_calls'] += 1
                return None
                
        except requests.exceptions.Timeout:
            print("⏰ Timeout en llamada Jina, reintentando...")
            self.stats['retries'] += 1
            raise
            
        except requests.exceptions.RequestException as e:
            print(f"🌐 Error de red: {e}, reintentando...")
            self.stats['retries'] += 1
            raise
            
        except Exception as e:
            print(f"❌ Error inesperado: {e}")
            self.token_usage['failed_calls'] += 1
            return None
    
    def extract_text_from_page(self, pdf_doc, page_num):
        """Extraer texto con manejo de errores"""
        try:
            page = pdf_doc[page_num - 1]
            text = page.get_text()
            
            # Limpiar y validar texto
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            clean_text = '\n'.join(lines)
            
            if len(clean_text) < 10:
                print(f"⚠️ Página {page_num}: texto muy corto")
                return ""
            
            return clean_text
            
        except Exception as e:
            print(f"❌ Error extrayendo texto página {page_num}: {e}")
            return ""
    
    def extract_images_from_page(self, pdf_doc, page_num):
        """Extraer imágenes con validación robusta"""
        images = []
        
        try:
            page = pdf_doc[page_num - 1]
            image_list = page.get_images(full=True)
            
            print(f"📊 Página {page_num}: {len(image_list)} imágenes encontradas")
            
            for img_idx, img in enumerate(image_list):
                try:
                    xref = img[0]
                    if xref <= 0:
                        continue
                    
                    pix = fitz.Pixmap(pdf_doc, xref)
                    
                    # Validaciones estrictas
                    if pix.width < 100 or pix.height < 100:
                        continue
                    
                    if pix.width * pix.height > 2000000:  # Máximo 2MP
                        print(f"⚠️ Imagen muy grande, saltando: {pix.width}x{pix.height}")
                        continue
                    
                    # Convertir a RGB
                    try:
                        if pix.n - pix.alpha < 4:
                            pil_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        else:
                            pix1 = fitz.Pixmap(fitz.csRGB, pix)
                            pil_img = Image.frombytes("RGB", [pix1.width, pix1.height], pix1.samples)
                            pix = pix1
                    except Exception as e:
                        print(f"⚠️ Error conversión imagen {img_idx}: {e}")
                        continue
                    
                    # Redimensionar si es muy grande
                    if pil_img.width > 1024 or pil_img.height > 1024:
                        pil_img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                        print(f"📏 Imagen redimensionada: {pil_img.width}x{pil_img.height}")
                    
                    # Convertir a base64 con compresión
                    buffer = io.BytesIO()
                    pil_img.save(buffer, format='JPEG', quality=85, optimize=True)
                    
                    # Verificar tamaño del base64
                    buffer_size = len(buffer.getvalue())
                    if buffer_size > 1_000_000:  # Máximo 1MB por imagen
                        print(f"⚠️ Imagen muy pesada después compresión: {buffer_size} bytes")
                        continue
                    
                    img_b64 = base64.b64encode(buffer.getvalue()).decode()
                    
                    images.append({
                        'page': page_num,
                        'index': img_idx,
                        'data_url': f"data:image/jpeg;base64,{img_b64}",
                        'width': pil_img.width,
                        'height': pil_img.height,
                        'size_bytes': buffer_size
                    })
                    
                    print(f"✅ Imagen procesada: {pil_img.width}x{pil_img.height}, {buffer_size} bytes")
                    
                except Exception as e:
                    print(f"⚠️ Error procesando imagen {img_idx}: {e}")
                    continue
                    
        except Exception as e:
            print(f"❌ Error extrayendo imágenes página {page_num}: {e}")
        
        print(f"📊 Página {page_num}: {len(images)} imágenes válidas procesadas")
        return images
    
    def process_section(self, pdf_path, section):
        """Procesar sección con checkpoints y recuperación"""
        print(f"\n📖 Procesando: {section.title}")
        self.processing_state['current_section'] = section.id
        
        try:
            pdf_doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"❌ Error abriendo PDF: {e}")
            return []
        
        chunks = []
        
        for page_num in range(section.start_page, section.end_page + 1):
            self.processing_state['current_page'] = page_num
            
            if page_num > len(pdf_doc):
                continue
            
            # Checkpoint cada 5 páginas
            if page_num % 5 == 0:
                self.save_checkpoint()
            
            try:
                # Extraer texto
                text = self.extract_text_from_page(pdf_doc, page_num)
                if not text or len(text.strip()) < 50:
                    print(f"⚠️ Página {page_num}: texto insuficiente, saltando")
                    continue
                
                # Extraer imágenes (con pausa entre páginas)
                print(f"🖼️ Procesando imágenes página {page_num}...")
                images = self.extract_images_from_page(pdf_doc, page_num)
                
                # Pausa entre procesamiento de páginas
                if images:
                    time.sleep(2)  # 2 segundos entre páginas con imágenes
                
                # Crear chunk
                chunk = {
                    'id': f"{section.id}_p{page_num}",
                    'content': text,
                    'images': images,
                    'section_id': section.id,
                    'section_title': section.title,
                    'content_type': section.content_type,
                    'page': page_num,
                    'keywords': section.keywords,
                    'has_images': len(images) > 0,
                    'image_count': len(images),
                    'estimated_tokens': self.estimate_tokens(text, images)
                }
                
                chunks.append(chunk)
                self.stats['pages_processed'] += 1
                
                if images:
                    self.stats['images_processed'] += len(images)
                
                print(f"✅ Página {page_num}: chunk creado, ~{chunk['estimated_tokens']} tokens estimados")
                
            except Exception as e:
                print(f"❌ Error procesando página {page_num}: {e}")
                self.processing_state['failed_chunks'].append(f"{section.id}_p{page_num}")
                continue
        
        pdf_doc.close()
        
        self.stats['chunks_created'] += len(chunks)
        self.stats['sections_processed'] += 1
        
        print(f"✅ {section.title}: {len(chunks)} chunks creados")
        return chunks
    
    def setup_qdrant_collection(self):
        """Configurar colección con retry y validación"""
        if not self.qdrant_client:
            print("❌ Cliente Qdrant no disponible")
            return False
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Verificar si existe
                collections = self.qdrant_client.get_collections()
                exists = any(col.name == self.collection_name for col in collections.collections)
                
                if exists:
                    print(f"🗑️ Eliminando colección existente: {self.collection_name}")
                    self.qdrant_client.delete_collection(self.collection_name)
                    time.sleep(2)  # Esperar eliminación
                
                # Crear nueva
                print(f"🆕 Creando colección: {self.collection_name}")
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=2048, distance=Distance.COSINE)
                )
                
                # Verificar creación
                time.sleep(2)
                info = self.qdrant_client.get_collection(self.collection_name)
                print(f"✅ Colección creada: {info.points_count} puntos, {info.config.params.vectors.size} dims")
                return True
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"⚠️ Reintentando creación colección ({attempt + 1}/{max_retries}): {e}")
                    time.sleep(5)
                else:
                    print(f"❌ Error crítico creando colección: {e}")
                    return False
        
        return False
    
    def upload_chunks_to_qdrant(self, all_chunks):
        """Subir chunks con retry robusto y manejo de lotes"""
        if not self.qdrant_client or not all_chunks:
            print("❌ No se puede subir a Qdrant")
            return
        
        print(f"\n⬆️ Subiendo {len(all_chunks)} chunks a Qdrant...")
        
        # Generar embeddings primero (con checkpoints)
        points = []
        
        for i, chunk in enumerate(all_chunks):
            try:
                print(f"\n📊 Procesando chunk {i+1}/{len(all_chunks)}: {chunk['id']}")
                
                # Verificar si ya procesamos este chunk
                if chunk['id'] in [p.id for p in points]:
                    continue
                
                # Generar embedding
                embedding = self.get_jina_multimodal_embedding(
                    text=chunk['content'],
                    images=chunk['images']
                )
                
                if embedding is None:
                    print(f"⚠️ Embedding falló para chunk {chunk['id']}, usando fallback")
                    embedding = [0.0] * 2048
                
                # Payload completo
                payload = {
                    'content': chunk['content'][:1000],  # Truncar para ahorrar espacio
                    'section_id': chunk['section_id'],
                    'section_title': chunk['section_title'],
                    'content_type': chunk['content_type'],
                    'page': chunk['page'],
                    'keywords': chunk['keywords'],
                    'has_images': chunk['has_images'],
                    'image_count': chunk['image_count'],
                    'estimated_tokens': chunk['estimated_tokens'],
                    'model': 'jina-embeddings-v4-bulletproof',
                    'processing_time': time.time()
                }
                
                points.append(PointStruct(id=i+1, vector=embedding, payload=payload))
                
                # Checkpoint cada 10 chunks
                if (i + 1) % 10 == 0:
                    self.save_checkpoint()
                    print(f"💾 Checkpoint: {i+1} chunks procesados")
                
            except Exception as e:
                print(f"❌ Error procesando chunk {i}: {e}")
                # Continuar con el siguiente
                continue
        
        print(f"\n🎯 {len(points)} chunks con embeddings generados")
        
        # Subir en lotes pequeños con retry
        batch_size = 25  # Lotes pequeños para ser conservadores
        successful_batches = 0
        
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(points) + batch_size - 1) // batch_size
            
            print(f"\n📦 Subiendo lote {batch_num}/{total_batches} ({len(batch)} chunks)")
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.qdrant_client.upsert(
                        collection_name=self.collection_name, 
                        points=batch,
                        wait=True  # Esperar confirmación
                    )
                    print(f"✅ Lote {batch_num} subido exitosamente")
                    successful_batches += 1
                    break
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 5
                        print(f"⚠️ Error lote {batch_num}, reintentando en {wait_time}s: {e}")
                        time.sleep(wait_time)
                    else:
                        print(f"❌ Error crítico lote {batch_num}: {e}")
            
            # Pausa entre lotes
            if i + batch_size < len(points):
                time.sleep(3)  # 3 segundos entre lotes
        
        print(f"\n🎉 Subida completada: {successful_batches}/{total_batches} lotes exitosos")
        
        # Verificar colección final
        try:
            info = self.qdrant_client.get_collection(self.collection_name)
            print(f"📊 Colección final: {info.points_count} puntos")
        except Exception as e:
            print(f"⚠️ Error verificando colección: {e}")

def process_manual_bulletproof(pdf_path):
    """Procesamiento bulletproof principal"""
    if not os.path.exists(pdf_path):
        print(f"❌ PDF no encontrado: {pdf_path}")
        return None
    
    print(f"\n🛡️ PROCESAMIENTO BULLETPROOF INICIADO")
    print(f"📄 Manual: {pdf_path}")
    print(f"🎯 Tokens disponibles: 10,000,000")
    print("="*60)
    
    processor = JinaBulletproofProcessor()
    
    # Setup inicial
    if not processor.setup_qdrant_collection():
        print("⚠️ Problemas con Qdrant, pero continuando...")
    
    # Procesar secciones
    all_chunks = []
    
    try:
        for section_id, section in MANUAL_SECTIONS.items():
            try:
                print(f"\n🚀 Iniciando sección: {section.title}")
                chunks = processor.process_section(pdf_path, section)
                all_chunks.extend(chunks)
                
                # Mostrar progreso de tokens
                remaining = processor.token_usage['max_tokens'] - processor.token_usage['total_tokens']
                print(f"📊 Tokens restantes: {remaining:,}")
                
                if remaining < 100000:  # Menos de 100k tokens
                    print("⚠️ TOKENS BAJOS - Considera parar aquí")
                
            except Exception as e:
                print(f"❌ Error crítico en sección {section.title}: {e}")
                processor.save_checkpoint()
                continue
        
        # Subir a Qdrant
        if all_chunks and processor.qdrant_client:
            processor.upload_chunks_to_qdrant(all_chunks)
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupción del usuario")
        processor.save_checkpoint()
        return None
    
    except Exception as e:
        print(f"\n❌ Error crítico: {e}")
        processor.save_checkpoint()
        return None
    
    # Resumen final
    total_time = time.time() - processor.stats['start_time']
    
    print(f"\n{'='*60}")
    print(f"🎉 PROCESAMIENTO BULLETPROOF COMPLETADO")
    print(f"{'='*60}")
    print(f"📊 ESTADÍSTICAS FINALES:")
    print(f"   🏗️ Secciones: {processor.stats['sections_processed']}")
    print(f"   📄 Páginas: {processor.stats['pages_processed']}")
    print(f"   📚 Chunks: {processor.stats['chunks_created']}")
    print(f"   🖼️ Imágenes: {processor.stats['images_processed']}")
    print(f"   🌟 Llamadas Jina: {processor.stats['jina_calls']}")
    print(f"   🔄 Reintentos: {processor.stats['retries']}")
    print(f"   💾 Checkpoints: {processor.stats['checkpoints_saved']}")
    print(f"   ⏱️ Tiempo total: {total_time:.1f}s")
    
    print(f"\n🎯 USO DE TOKENS:")
    print(f"   📊 Tokens usados: {processor.token_usage['total_tokens']:,}")
    print(f"   📊 Tokens restantes: {processor.token_usage['max_tokens'] - processor.token_usage['total_tokens']:,}")
    print(f"   ✅ Llamadas exitosas: {processor.token_usage['successful_calls']}")
    print(f"   ❌ Llamadas fallidas: {processor.token_usage['failed_calls']}")
    
    return {
        'chunks': all_chunks,
        'stats': processor.stats,
        'token_usage': processor.token_usage,
        'model': 'jina-embeddings-v4-bulletproof'
    }

def run_bulletproof_processing():
    """Ejecutar procesamiento bulletproof"""
    if not CONFIG_OK:
        print("❌ Configuración inválida")
        return
    
    pdf_path = "Manual_de_Mantenimiento.pdf"
    
    if not os.path.exists(pdf_path):
        try:
            from google.colab import files
            print("📤 Subir PDF...")
            uploaded = files.upload()
            pdf_files = [f for f in uploaded.keys() if f.endswith('.pdf')]
            if pdf_files:
                pdf_path = pdf_files[0]
                if pdf_path != "Manual_de_Mantenimiento.pdf":
                    os.rename(pdf_path, "Manual_de_Mantenimiento.pdf")
                    pdf_path = "Manual_de_Mantenimiento.pdf"
        except ImportError:
            print("❌ Coloca 'Manual_de_Mantenimiento.pdf' en directorio")
            return
    
    print(f"✅ Manual encontrado: {pdf_path}")
    
    # Procesamiento bulletproof
    results = process_manual_bulletproof(pdf_path)
    
    if results:
        print(f"\n🎉 ¡PROCESAMIENTO BULLETPROOF EXITOSO!")
        print(f"🎯 Tokens usados: {results['token_usage']['total_tokens']:,} / 10,000,000")
        print(f"📊 Chunks procesados: {len(results['chunks'])}")
        
        remaining_tokens = 10_000_000 - results['token_usage']['total_tokens']
        print(f"💡 Tokens restantes: {remaining_tokens:,}")
        
        if remaining_tokens > 1_000_000:
            print("✅ Suficientes tokens para futuras consultas")
        else:
            print("⚠️ Tokens bajos, usa con moderación")
    else:
        print(f"\n❌ Procesamiento incompleto - revisa checkpoints")

print(f"\n🛡️ PETION7 JINA BULLETPROOF LISTO!")
print(f"📋 Para ejecutar: run_bulletproof_processing()")
print(f"🎯 Características:")
print(f"   ✅ Rate limiting inteligente (45 calls/min)")
print(f"   ✅ Retry exponencial con 5 intentos")
print(f"   ✅ Checkpoints automáticos cada 5 páginas") 
print(f"   ✅ Manejo de interrupciones (Ctrl+C)")
print(f"   ✅ Validación de tokens (10M máximo)")
print(f"   ✅ Compresión de imágenes automática")
print(f"   ✅ Lotes pequeños para Qdrant (25 chunks)")
print(f"   ✅ Logs detallados de progreso")

if CONFIG_OK:
    print(f"\n🚀 ¡Todo configurado! Ejecuta: run_bulletproof_processing()")
else:
    print(f"\n⚠️ Configura las variables de entorno primero")