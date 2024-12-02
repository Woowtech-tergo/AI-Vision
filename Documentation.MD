# Counter Project Documentation

## Introdução

O **Counter Project** é um projeto voltado para o desenvolvimento de uma interface interativa utilizando **Gradio** e a implementação de modelos de inteligência artificial focados em soluções de **visão computacional**. Este sistema é projetado como uma **prova de conceito (PoC)**, destinada a ser avaliada pela equipe de usuários internos para testar suas funcionalidades e potencialidades. O objetivo final é criar uma PoC funcional e útil que possa ser apresentada a futuros clientes interessados em nossa tecnologia.

## Objetivo

Desenvolver uma plataforma modular e escalável que permita a integração de diversos modelos de detecção de objetos e pessoas, aplicando técnicas avançadas de visão computacional para resolver problemas reais em diferentes setores. A interface deve ser intuitiva e acessível, facilitando a interação do usuário com os modelos de inteligência artificial implementados.

## Tecnologias Utilizadas

### Gradio

[Gradio](https://gradio.app/) é uma biblioteca Python que facilita a criação de interfaces de usuário personalizadas para aplicações de machine learning. Com Gradio, podemos:

- Construir interfaces web interativas com pouco código.
- Permitir que usuários interajam com modelos de IA em tempo real.
- Compartilhar facilmente aplicações via links públicos.

**Execução da Interface:**

Para executar a interface Gradio, utilize o seguinte comando:

```bash
python app.py
```

### OpenCV

[OpenCV](https://opencv.org/) (Open Source Computer Vision Library) é uma biblioteca de código aberto que fornece centenas de algoritmos otimizados para processamento de imagens e visão computacional.

- **Aplicações:** Detecção de objetos, reconhecimento facial, segmentação de imagens, entre outros.
- **Vantagens:** Alto desempenho, extensa documentação e grande comunidade de usuários.

### YOLO

[YOLO](https://pjreddie.com/darknet/yolo/) (You Only Look Once) é uma família de modelos de detecção de objetos em tempo real.

- **Características:**
  - Alta velocidade de processamento.
  - Detecção de múltiplas classes de objetos.
  - Implementação eficiente em diversos dispositivos.

### Deep Sort

[Deep SORT](https://arxiv.org/abs/1703.07402) (Simple Online and Realtime Tracking with a Deep Association Metric) é um algoritmo avançado para rastreamento de múltiplos objetos.

- **Funcionamento:**
  - Combina informações de detecção (como as fornecidas pelo YOLO) com métricas de associação baseadas em aprendizado profundo.
  - Mantém a identidade de objetos individuais ao longo de sequências de vídeo.

- **Vantagens:**
  - Alta precisão no rastreamento.
  - Capacidade de lidar com oclusões e desaparecimentos temporários.

## Arquitetura Pretendida

O projeto está organizado de forma modular, permitindo a fácil integração e substituição de componentes. A estrutura básica é a seguinte:

- **Interface Gradio:** Responsável pela interação com o usuário, permitindo o upload de vídeos, seleção de modelos e visualização dos resultados.
- **Camada de Processamento:** Onde os modelos de visão computacional (YOLO, Deep Sort, etc.) são executados.
- **Pasta `Modelos`:** Contém os modelos de detecção atuais e futuros. Pretendemos expandir esta pasta com:

  - Novos modelos de detecção de objetos.
  - Modelos específicos para contagem de pessoas.
  - Aplicações personalizadas para diferentes cenários (segurança, varejo, etc.).

- **Camada de Serviços:** Planejamos implementar APIs e serviços que permitam a integração com outras plataformas e sistemas.

## Desenvolvimento e Implantação

Atualmente, o desenvolvimento em produção está hospedado no servidor **Railway**, proporcionando um ambiente escalável e confiável para a execução da aplicação.

**Link para Acesso:**

[Counter Project em Produção](https://counterproject-production.up.railway.app/)

## Instruções de Execução

1. **Pré-requisitos:**

   - Python 3.8 ou superior.
   - Bibliotecas listadas no arquivo `requirements.txt`.

2. **Clonar o Repositório:**

   ```bash
   https://github.com/gustavofelicidade/ConuterProject.git
   ```

3. **Instalar Dependências:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Executar a Aplicação:**

   ```bash
   python app.py
   ```

5. **Acessar a Interface:**

   - Acesse `http://localhost:7860` no seu navegador.

## Futuras Implementações

- **Expansão de Modelos:**
  - Adicionar modelos treinados para detecção de diferentes objetos e cenários.
  - Incorporar técnicas de segmentação semântica e instance segmentation.

- **Melhorias na Interface:**
  - Fornecer feedback em tempo real sobre o processamento.
  - Permitir ajustes finos nos parâmetros dos modelos diretamente pela interface.

- **Integração com Bancos de Dados:**
  - Armazenar resultados e métricas para análise posterior.
  - Implementar sistemas de alerta e notificações.

- **Desenvolvimento de APIs:**
  - Facilitar a integração com sistemas externos.

## Conclusão

O **Counter Project** representa um passo significativo em nossa jornada para oferecer soluções de visão computacional avançadas e acessíveis. A colaboração contínua e o feedback da equipe serão fundamentais para refinar e expandir este projeto.

---

**Contato para Mais Informações:**

- **Desenvolvedor Responsável:** Gustavo Felicidade
- **Email:** gustavo.costa@tergo.com
