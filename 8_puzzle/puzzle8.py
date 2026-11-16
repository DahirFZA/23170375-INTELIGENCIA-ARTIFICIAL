import time
import heapq
from collections import deque

META = (1, 2, 3, 4, 5, 6, 7, 8, 0)


def obtener_vecinos(estado):
    """Genera los estados sucesores moviendo el espacio vacío (0)."""
    vecinos = []
    idx = estado.index(0)
    fila, col = divmod(idx, 3)

    movimientos = {
        "Arriba": (-1, 0),
        "Abajo": (1, 0),
        "Izquierda": (0, -1),
        "Derecha": (0, 1),
    }

    for accion, (df, dc) in movimientos.items():
        nf, nc = fila + df, col + dc
        if 0 <= nf < 3 and 0 <= nc < 3:
            n_idx = nf * 3 + nc
            lista = list(estado)
            lista[idx], lista[n_idx] = lista[n_idx], lista[idx]
            vecinos.append((accion, tuple(lista)))
    return vecinos


def manhattan(estado):
    """Heurística de distancia Manhattan hacia el estado META."""
    distancia = 0
    for i, valor in enumerate(estado):
        if valor != 0:
            fila_actual, col_actual = divmod(i, 3)
            fila_meta, col_meta = divmod(valor - 1, 3)
            distancia += abs(fila_actual - fila_meta) + abs(col_actual - col_meta)
    return distancia


def resolver_bfs(inicial):
    inicio_t = time.perf_counter()
    cola = deque([(inicial, [])])
    visitados = {inicial}
    nodos_explorados = 0

    while cola:
        estado, camino = cola.popleft()
        nodos_explorados += 1

        if estado == META:
            tiempo_ms = (time.perf_counter() - inicio_t) * 1000
            return camino, nodos_explorados, tiempo_ms

        for accion, vecino in obtener_vecinos(estado):
            if vecino not in visitados:
                visitados.add(vecino)
                cola.append((vecino, camino + [accion]))

    return None, nodos_explorados, 0.0


def resolver_dfs(inicial, limite=25):
    inicio_t = time.perf_counter()
    pila = [(inicial, [], {inicial})]
    nodos_explorados = 0

    while pila:
        estado, camino, ruta_visitada = pila.pop()
        nodos_explorados += 1

        if estado == META:
            tiempo_ms = (time.perf_counter() - inicio_t) * 1000
            return camino, nodos_explorados, tiempo_ms

        if len(camino) < limite:
            for accion, vecino in reversed(obtener_vecinos(estado)):
                if vecino not in ruta_visitada:
                    pila.append((vecino, camino + [accion], ruta_visitada | {vecino}))

    tiempo_ms = (time.perf_counter() - inicio_t) * 1000
    return None, nodos_explorados, tiempo_ms


def resolver_a_estrella(inicial):
    inicio_t = time.perf_counter()
    frontera = [(manhattan(inicial), 0, inicial, [])]
    mejor_costo = {inicial: 0}
    nodos_explorados = 0

    while frontera:
        _, g, estado, camino = heapq.heappop(frontera)
        nodos_explorados += 1

        if estado == META:
            tiempo_ms = (time.perf_counter() - inicio_t) * 1000
            return camino, nodos_explorados, tiempo_ms

        for accion, vecino in obtener_vecinos(estado):
            nuevo_g = g + 1
            if vecino not in mejor_costo or nuevo_g < mejor_costo[vecino]:
                mejor_costo[vecino] = nuevo_g
                f = nuevo_g + manhattan(vecino)
                heapq.heappush(frontera, (f, nuevo_g, vecino, camino + [accion]))

    return None, nodos_explorados, 0.0


def imprimir_tablero(estado):
    for i in range(0, 9, 3):
        fila = [str(x) if x != 0 else " " for x in estado[i : i + 3]]
        print(" | ".join(fila))
    print("-" * 9)


if __name__ == "__main__":
    # Estado inicial resoluble (a 8 movimientos de la meta)
    estado_inicial = (1, 2, 3, 4, 0, 6, 7, 5, 8)

    print("Estado inicial:")
    imprimir_tablero(estado_inicial)

    resultados = []
    for nombre, funcion in [
        ("BFS (Anchura)", resolver_bfs),
        ("DFS (Profundidad)", resolver_dfs),
        ("A* (Manhattan)", resolver_a_estrella),
    ]:
        camino, nodos, tiempo = funcion(estado_inicial)
        pasos = len(camino) if camino is not None else "Sin solución"
        resultados.append((nombre, pasos, nodos, f"{tiempo:.3f} ms"))
        if nombre == "A* (Manhattan)" and camino:
            print(f"Camino óptimo encontrado ({pasos} pasos): {camino}\n")

    encabezado = f"{'Algoritmo':<20} | {'Pasos':<12} | {'Nodos explorados':<18} | {'Tiempo':<12}"
    separador = "-" * len(encabezado)
    lineas_tabla = [encabezado, separador]

    for alg, pasos, nodos, tiempo in resultados:
        lineas_tabla.append(f"{alg:<20} | {str(pasos):<12} | {str(nodos):<18} | {tiempo:<12}")

    texto_tabla = "\n".join(lineas_tabla)
    print(texto_tabla)

    with open("8_puzzle/tabla.txt", "w", encoding="utf-8") as f:
        f.write("TABLA COMPARATIVA - 8 PUZZLE\n")
        f.write(f"Estado inicial: {estado_inicial}\n\n")
        f.write(texto_tabla + "\n")
    print("\nResultados guardados en 8_puzzle/tabla.txt")