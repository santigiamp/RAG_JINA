FROM python:3.10-slim

# Seteo de directorio
WORKDIR /app

# Copiamos requerimientos
COPY requirements.txt .

# Instalamos dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el resto del código
COPY . .

# Exponemos el puerto
EXPOSE 7860

# Comando para iniciar Langflow
CMD ["langflow", "run", "--host", "0.0.0.0", "--port", "7860"]
