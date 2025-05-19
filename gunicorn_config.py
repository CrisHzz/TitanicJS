import multiprocessing

# Configuración de workers optimizada para bajo consumo de memoria
workers = 1  # Solo un worker para evitar duplicar la carga del modelo
threads = 2  # Usar threads para manejar múltiples solicitudes
worker_class = 'sync'  # Clase de worker síncrono es más ligera

# Timeouts más altos para evitar que se maten procesos durante la carga del modelo
timeout = 120
keepalive = 5

# Limitar el uso de memoria
max_requests = 100  # Reiniciar workers después de 100 requests para evitar memory leaks
max_requests_jitter = 10  # Añadir aleatoriedad para evitar que todos los workers se reinicien a la vez

# Configuración de logging
loglevel = 'error'
accesslog = '-'
errorlog = '-'

# Manejo de errores
capture_output = True
enable_stdio_inheritance = True 