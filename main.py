import os
from interface.app import app

if __name__ == "__main__":
    # Use a porta fornecida pela variável de ambiente PORT
    port = int(os.getenv("PORT", 7860))
    app.launch(server_name="0.0.0.0", server_port=port)
