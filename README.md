# Gridworld RL – Evaluación y Mejora de Políticas

Este repositorio contiene una implementación educativa en Python para estudiar **Evaluación de Políticas**, **Mejora de Políticas** y **Value Iteration** dentro de un entorno tipo **Gridworld 4x4**, uno de los ejemplos clásicos de Aprendizaje por Refuerzo (Reinforcement Learning).

El objetivo es visualizar cómo evolucionan las funciones de valor y las políticas a medida que se aplican iterativamente las ecuaciones de Bellman.

---

## 📌 Contenido

El código incluye:

### ✔ Definición del entorno 4x4
- 16 estados numerados del 0 al 15.
- Dos estados terminales: **0** y **15**.
- Recompensa de **–1** en cada transición.
- Transiciones deterministas para 4 acciones:  
  `0=arriba`, `1=abajo`, `2=derecha`, `3=izquierda`.

### ✔ Visualización del entorno
La función `plot_env()` dibuja:
- La matriz de valores `v(s)`
- Las flechas de la política `π(s)`

Este gráfico permite ver de forma intuitiva la mejora de la política.

🧩 Requisitos

Antes de ejecutar el script, instala las dependencias:

pip install -r requirements.txt

🧑‍💻 Autor

Desarrollado por Gus como parte de su aprendizaje en Python e IA.
