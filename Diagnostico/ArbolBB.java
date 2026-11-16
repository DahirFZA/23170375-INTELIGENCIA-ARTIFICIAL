public class ArbolBB {
    private NodoABB raiz;

    public ArbolBB() {
        this.raiz = null;
    }

    public boolean estaVacio() {
        return raiz == null;
    }

    public void insertar(int dato) {
        raiz = insertarRec(raiz, dato);
    }

    private NodoABB insertarRec(NodoABB actual, int dato) {
        if (actual == null) {
            return new NodoABB(dato);
        }
        if (dato < actual.dato) {
            actual.izquierdo = insertarRec(actual.izquierdo, dato);
        } else if (dato > actual.dato) {
            actual.derecho = insertarRec(actual.derecho, dato);
        }
        return actual;
    }

    public boolean buscar(int dato) {
        return buscarRec(raiz, dato);
    }

    private boolean buscarRec(NodoABB actual, int dato) {
        if (actual == null) return false;
        if (dato == actual.dato) return true;
        return dato < actual.dato
                ? buscarRec(actual.izquierdo, dato)
                : buscarRec(actual.derecho, dato);
    }

    public void imprimirInorden() {
        System.out.print("Inorden:   ");
        inordenRec(raiz);
        System.out.println();
    }

    private void inordenRec(NodoABB actual) {
        if (actual != null) {
            inordenRec(actual.izquierdo);
            System.out.print(actual.dato + " ");
            inordenRec(actual.derecho);
        }
    }

    public void imprimirPreorden() {
        System.out.print("Preorden:  ");
        preordenRec(raiz);
        System.out.println();
    }

    private void preordenRec(NodoABB actual) {
        if (actual != null) {
            System.out.print(actual.dato + " ");
            preordenRec(actual.izquierdo);
            preordenRec(actual.derecho);
        }
    }

    public void imprimirPostorden() {
        System.out.print("Postorden: ");
        postordenRec(raiz);
        System.out.println();
    }

    private void postordenRec(NodoABB actual) {
        if (actual != null) {
            postordenRec(actual.izquierdo);
            postordenRec(actual.derecho);
            System.out.print(actual.dato + " ");
        }
    }

    public static void main(String[] args) {
        ArbolBB arbol = new ArbolBB();
        int[] valores = {50, 30, 70, 20, 40, 60, 80};

        for (int v : valores) {
            arbol.insertar(v);
        }

        arbol.imprimirInorden();
        arbol.imprimirPreorden();
        arbol.imprimirPostorden();

        System.out.println("¿Existe el 40?: " + arbol.buscar(40));
        System.out.println("¿Existe el 99?: " + arbol.buscar(99));
    }
}