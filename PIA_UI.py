import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
import time
import threading
import weakref

# ---------------------- Heurística ----------------------
def select_node(graph, rewards, current, visited):
    best_node = None
    best_ratio = -1
    for j in range(len(graph)):
        if j not in visited:
            ratio = rewards[j] / graph[current][j]
            if ratio > best_ratio:
                best_ratio = ratio
                best_node = j
    return best_node

def heuristic(graph, rewards, start, T_max, update_callback=None, get_delay=lambda: 1.0, self_ref=lambda: None):
    route = [start]
    current_time = 0
    total_reward = 0
    visited = {start}
    summary = []

    while current_time < T_max:
        next_node = select_node(graph, rewards, route[-1], visited)
        if next_node is None:
            break

        d = graph[route[-1]][next_node]
        return_time = graph[next_node][start]

        if current_time + d + return_time <= T_max:
            route.append(next_node)
            visited.add(next_node)
            current_time += d
            total_reward += rewards[next_node]
            summary.append(rewards[next_node])

            if update_callback:
                update_callback(route)
                while self_ref() and self_ref().paused:
                    time.sleep(0.1)
                time.sleep(get_delay())
        else:
            current_time += graph[route[-1]][start]
            break

    route.append(start)
    if update_callback:
        update_callback(route)

    return route, total_reward, current_time, summary


# ---------------------- Leer archivo ----------------------
def reading_file(filename):
    with open(filename, 'r') as file:
        data = file.readlines()

    data = [line.strip() for line in data]
    index = 0

    n = int(data[index])
    index += 1

    graph = []
    for _ in range(n):
        graph.append(list(map(int, data[index].split())))
        index += 1
    graph = np.array(graph)

    rewards = list(map(int, data[index].split()))
    index += 1

    start = int(data[index])
    index += 1
    T_max = int(data[index])

    return graph, rewards, start, T_max


# ---------------------- Interfaz gráfica ----------------------
class GraphApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Heurística de Recorrido")

        self.canvas = tk.Canvas(root, width=600, height=600, bg="white")
        self.canvas.pack()

        self.speed_combo = ttk.Combobox(root, values=["x0.5", "x1", "x5", "x10", "x20"])
        self.speed_combo.set("x1")
        self.speed_combo.pack()

        self.speed_values = {"x0.5": 2.0, "x1": 1.0, "x5": 0.2, "x10": 0.1, "x20": 0.05}

        self.load_button = ttk.Button(root, text="📂 Cargar instancia", command=self.load_instance)
        self.load_button.pack()

        self.replay_button = ttk.Button(root, text="🔁 Repetir recorrido", command=self.replay)
        self.replay_button.pack()

        self.pause_button = ttk.Button(root, text="⏸ Pausar", command=self.toggle_pause)
        self.pause_button.pack()

        self.result_label = tk.Label(root, text="")
        self.result_label.pack()

        self.route_label = tk.Label(root, text="Recorrido: ")
        self.route_label.pack()

        self.graph = None
        self.rewards = None
        self.start = None
        self.T_max = None
        self.last_route = []
        self.paused = False

        self.node_positions = []  # Guardar posiciones de los nodos
        self.node_objects = []  # Guardar objetos de nodos en el lienzo

    def draw_graph(self, route):
        if not self.node_positions:
            # Solo calcular las posiciones una vez
            n = len(self.graph)
            self.node_positions = [(300 + 200 * np.cos(2 * np.pi * i / n), 300 + 200 * np.sin(2 * np.pi * i / n)) for i in range(n)]
            for i, (x, y) in enumerate(self.node_positions):
                # Guardar los nodos solo una vez
                node = self.canvas.create_oval(x-20, y-20, x+20, y+20, fill="lightblue")
                self.canvas.create_text(x, y, text=str(i))
                self.node_objects.append(node)

        # Dibujar solo el recorrido
        for i in range(len(route) - 1):
            a = self.node_positions[route[i]]
            b = self.node_positions[route[i + 1]]
            self.canvas.create_line(a[0], a[1], b[0], b[1], fill="red", width=2)

    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_button.config(text="▶ Reanudar" if self.paused else "⏸ Pausar")

    def load_instance(self):
        file_path = filedialog.askopenfilename()
        if not file_path:
            return

        self.graph, self.rewards, self.start, self.T_max = reading_file(file_path)
        self.last_route = []
        self.canvas.delete("all")
        self.node_positions = []  # Reset posiciones de nodos
        self.node_objects = []  # Reset nodos en el lienzo

        def update(route):
            self.canvas.delete("line")  # Solo eliminar las rutas, no los nodos
            self.draw_graph(route)
            self.update_route_label(route)  # Actualizar el recorrido

        def run_heuristic():
            self_ref = weakref.ref(self)
            route, total_reward, time_spent, summary = heuristic(
                self.graph,
                self.rewards,
                self.start,
                self.T_max,
                update_callback=update,
                get_delay=lambda: self.speed_values.get(self.speed_combo.get(), 1.0),
                self_ref=self_ref
            )
            self.last_route = route
            self.result_label.config(text=f"Total profit: {total_reward}, Time spent: {time_spent}, Summary: {summary}")

        threading.Thread(target=run_heuristic, daemon=True).start()

    def replay(self):
        if not self.last_route:
            return

        # Limpiar el lienzo antes de comenzar el recorrido
        self.canvas.delete("all")
        self.node_positions = []  # Reset posiciones de nodos
        self.node_objects = []  # Reset nodos en el lienzo

        def update(route):
            self.canvas.delete("line")  # Solo eliminar las rutas, no los nodos
            self.draw_graph(route)
            self.update_route_label(route)  # Actualizar el recorrido

        def run_replay():
            self_ref = weakref.ref(self)
            for i in range(1, len(self.last_route) + 1):
                update(self.last_route[:i])
                while self_ref() and self_ref().paused:
                    time.sleep(0.1)
                time.sleep(self.speed_values.get(self.speed_combo.get(), 1.0))

        threading.Thread(target=run_replay, daemon=True).start()

    def update_route_label(self, route):
        # Mostrar el recorrido como un arreglo, no como "->"
        self.route_label.config(text="Recorrido: " + str(route))


# ---------------------- Ejecutar app ----------------------
if __name__ == '__main__':
    root = tk.Tk()
    app = GraphApp(root)
    root.mainloop()
