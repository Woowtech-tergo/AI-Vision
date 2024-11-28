import os
from interface import app

if __name__ == "__main__":
    # Use a porta fornecida pela variável de ambiente PORT, se disponível
    port = int(os.getenv("PORT", 7860))
    # Lança a aplicação definida no app.py
    app.launch(server_name="0.0.0.0", server_port=port)
