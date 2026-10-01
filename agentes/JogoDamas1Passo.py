import copy

# 1. INICIALIZAÇÃO DO ESTADO DO JOGO
def inicializar_tabuleiro():
    # Criar tabuleiro 8x8 vazio (' ')
    tabuleiro = [[' ' for _ in range(8)] for _ in range(8)]
    
    # Adicionar peças Pretas ('P') no topo (linhas 0 a 2) nas casas escuras
    for r in range(3):
        for c in range(8):
            if (r + c) % 2 == 1:
                tabuleiro[r][c] = 'P'
                
    # Adicionar peças Brancas ('B') na base (linhas 5 a 7) nas casas escuras
    for r in range(5, 8):
        for c in range(8):
            if (r + c) % 2 == 1:
                tabuleiro[r][c] = 'B'
                
    return tabuleiro

def imprimir_tabuleiro(tabuleiro):
    print("\n   0   1   2   3   4   5   6   7")
    print(" ┌" + "───┬" * 7 + "───┐")
    for r in range(8):
        linha = f"{r}│ " + " │ ".join(tabuleiro[r]) + " │"
        print(linha)
        if r < 7:
            print(" ├" + "───┼" * 7 + "───┤")
    print(" └" + "───┴" * 7 + "───┘")

# 2. FUNÇÃO DE UTILIDADE / AVALIAÇÃO ESTÁTICA
def avaliar_tabuleiro(tabuleiro):
    """
    As Brancas tentam MAXIMIZAR este valor.
    As Pretas tentam MINIMIZAR este valor.
    """
    brancas = 0
    pretas = 0
    for r in range(8):
        for c in range(8):
            if tabuleiro[r][c] == 'B':
                brancas += 1
            elif tabuleiro[r][c] == 'P':
                pretas += 1
    return brancas - pretas

# 3. REGRAS DE TRANSIÇÃO (Geração de Jogadas Válidas)
def obter_movimentos_validos(tabuleiro, jogador):
    movimentos = []
    # Brancas sobem o tabuleiro (linha - 1), Pretas descem (linha + 1)
    direcao = -1 if jogador == 'B' else 1  
    
    for r in range(8):
        for c in range(8):
            if tabuleiro[r][c] == jogador:
                # Verificar diagonais à frente (esquerda e direita)
                for dc in [-1, 1]:
                    nr, nc = r + direcao, c + dc
                    if 0 <= nr < 8 and 0 <= nc < 8:
                        if tabuleiro[nr][nc] == ' ':
                            # Movimento válido: guarda (linha_origem, col_origem, linha_destino, col_destino)
                            movimentos.append((r, c, nr, nc))
    return movimentos

def mover_peca(tabuleiro, movimento):
    # Retorna um novo estado (tabuleiro) sem alterar o original
    tab_copia = copy.deepcopy(tabuleiro)
    ri, ci, rf, cf = movimento
    tab_copia[rf][cf] = tab_copia[ri][ci]
    tab_copia[ri][ci] = ' '
    return tab_copia

# 4. O ALGORITMO MINIMAX
def minimax(tabuleiro, profundidade, maximizando):
    # Caso base: Profundidade limite atingida
    if profundidade == 0:
        return avaliar_tabuleiro(tabuleiro), None

    if maximizando:
        maior_valor = float('-inf')
        melhor_movimento = None
        movimentos = obter_movimentos_validos(tabuleiro, 'B')
        
        if not movimentos: # Sem movimentos válidos (Fim de jogo)
            return avaliar_tabuleiro(tabuleiro), None
            
        for mov in movimentos:
            proximo_estado = mover_peca(tabuleiro, mov)
            valor, _ = minimax(proximo_estado, profundidade - 1, False)
            if valor > maior_valor:
                maior_valor = valor
                melhor_movimento = mov
        return maior_valor, melhor_movimento

    else:
        menor_valor = float('inf')
        melhor_movimento = None
        movimentos = obter_movimentos_validos(tabuleiro, 'P')
        
        if not movimentos: # Sem movimentos válidos (Fim de jogo)
            return avaliar_tabuleiro(tabuleiro), None
            
        for mov in movimentos:
            proximo_estado = mover_peca(tabuleiro, mov)
            valor, _ = minimax(proximo_estado, profundidade - 1, True)
            if valor < menor_valor:
                menor_valor = valor
                melhor_movimento = mov
        return menor_valor, melhor_movimento

# --- SIMULAÇÃO DA JOGADA ---
if __name__ == "__main__":
    tabuleiro_jogo = inicializar_tabuleiro()
    
    print("=== ESTADO INICIAL DO TABULEIRO ===")
    imprimir_tabuleiro(tabuleiro_jogo)
    print(f"Avaliação Inicial: {avaliar_tabuleiro(tabuleiro_jogo)}")

    # IA (Jogador MAX - Brancas) calcula a melhor jogada olhando 3 níveis à frente
    print("\n[IA Pensando...] Calculando melhor jogada para as Brancas ('B') com MiniMax (Profundidade = 3)...")
    valor, melhor_jogada = minimax(tabuleiro_jogo, profundidade=3, maximizando=True)
    
    if melhor_jogada:
        ri, ci, rf, cf = melhor_jogada
        print(f"\n> A IA escolheu mover a peça de ({ri}, {ci}) para ({rf}, {cf})")
        print(f"> Valor esperado da utilidade após esta árvore: {valor}")
        
        # Aplicar a jogada escolhida pela busca competitiva
        tabuleiro_jogo = mover_peca(tabuleiro_jogo, melhor_jogada)
        imprimir_tabuleiro(tabuleiro_jogo)
    else:
        print("Nenhuma jogada disponível.")