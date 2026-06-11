#!/bin/bash
# EC2 User Data - MeliExpert Chatbot
# Ejecutar al lanzar una instancia Amazon Linux 2023

set -e

PROJECT_DIR="/home/ec2-user/meliexpert"
APP_PORT=8501

# 1. Actualizar sistema
dnf update -y

# 2. Instalar Python y git
dnf install -y python3.11 python3.11-pip python3.11-devel git

# 3. Crear directorio del proyecto
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# 4. Crear entorno virtual
python3.11 -m venv .venv
source .venv/bin/activate

# 5. Copiar archivos (desde el mismo directorio o S3)
# Opcion A: Clonar repositorio (descomenta y edita)
# git clone https://github.com/TU_USUARIO/TU_REPO.git .

# Opcion B: Los archivos se subiran manualmente con scp
#   scp -i tu-key.pem app.py .env requirements.txt ec2-user@IP:/home/ec2-user/meliexpert/

# 6. Instalar dependencias
pip install --no-cache-dir -r requirements.txt

# 7. Crear archivo .env (EDITALO DESPUES con tus credenciales)
if [ ! -f .env ]; then
    cat > .env << 'EOF'
GITHUB_TOKEN=
CONFIG_EMAIL_REMITENTE=
CONFIG_EMAIL_PASSWORD=
OPENAI_BASE_URL=https://models.inference.ai.azure.com
GITHUB_BASE_URL=https://models.inference.ai.azure.com
OPENAI_EMBEDDINGS_URL=https://models.github.ai/inference
EOF
fi

# 8. Crear servicio systemd para que la app corra siempre
sudo tee /etc/systemd/system/meliexpert.service > /dev/null << EOF
[Unit]
Description=MeliExpert Chatbot
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=$PROJECT_DIR
Environment=PATH=$PROJECT_DIR/.venv/bin:/usr/bin
ExecStart=$PROJECT_DIR/.venv/bin/streamlit run app.py --server.port $APP_PORT --server.address 0.0.0.0
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# 9. Habilitar e iniciar servicio
sudo systemctl daemon-reload
sudo systemctl enable meliexpert
sudo systemctl start meliexpert

# 10. Abrir puerto en firewall (Amazon Linux 2023)
sudo dnf install -y firewalld
sudo systemctl start firewalld
sudo firewall-cmd --permanent --add-port=${APP_PORT}/tcp
sudo firewall-cmd --reload

echo "=== Instalacion completada ==="
echo "Accede en: http://$(curl -s http://checkip.amazonaws.com):$APP_PORT"
echo "No olvides editar .env con tu GITHUB_TOKEN"
echo "  sudo vi $PROJECT_DIR/.env"
echo "  sudo systemctl restart meliexpert"
