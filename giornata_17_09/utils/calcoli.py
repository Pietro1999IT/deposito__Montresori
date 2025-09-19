def somma(a, b) -> int:
    """
    Calcola la somma di due numeri.

    Parameters:
    a (int o float): Primo addendo.
    b (int o float): Secondo addendo.

    Returns:
    int: La somma di a e b.
    """
    return a + b

def conta_unici(lista) -> int:
    """
    Conta il numero di elementi unici presenti in una lista.

    Parameters:
    lista (list): Una lista.

    Returns:
    int: Numero di elementi unici nella lista.
    """
    return len(set(lista))
