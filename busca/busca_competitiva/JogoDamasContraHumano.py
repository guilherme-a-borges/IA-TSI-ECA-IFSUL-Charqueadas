import copy

# 1. INICIALIZAÇÃO DO ESTADO DO JOGO
def inicializar_tabuleiro():
    tabuleiro = [[' ' for _ in range(8)] for _ in range(8)]
    # Pretas ('P') no topo (linhas 0 a 2)
    for r in range(3):
        for c in range(8):
            if (r + c) % 2 == 1:
                tabuleiro[r][c] = 'P'
    # Brancas ('B') na base (linhas 5 a 7)
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

# 2. FUNÇÃO DE AVALIAÇÃO ESTÁTICA (Soma Zero)
def avaliar_tabuleiro(tabuleiro):
    brancas = 0
    pretas = 0
    for r in range(8):
        for c in range(8):
            if tabuleiro[r][c] == 'B':
                brancas += 1
            elif tabuleiro[r][c] == 'P':
                pretas += 1
    return brancas - pretas

# 3. REGRAS DE TRANSIÇÃO
def obter_movimentos_validos(tabuleiro, jogador):
    movimentos = []
    direcao = -1 if jogador == 'B' else 1  # 'B' sobe, 'P' desce
    
    for r in range(8):
        for c in range(8):
            if tabuleiro[r][c] == jogador:
                for dc in [-1, 1]:
                    nr, nc = r + direcao, c + dc
                    if 0 <= nr < 8 and 0 <= nc < 8:
                        if tabuleiro[nr][nc] == ' ':
                            movimentos.append((r, c, nr, nc))
    return movimentos

def mover_peca(tabuleiro, movimento):
    tab_copia = copy.deepcopy(tabuleiro)
    ri, ci, rf, cf = movimento
    tab_copia[rf][cf] = tab_copia[ri][ci]
    tab_copia[ri][ci] = ' '
    return tab_copia

# 4. O ALGORITMO MINIMAX
def minimax(tabuleiro, profundidade, maximizando):
    if profundidade == 0:
        return avaliar_tabuleiro(tabuleiro), None

    if maximizando:
        maior_valor = float('-inf')
        melhor_movimento = None
        movimentos = obter_movimentos_validos(tabuleiro, 'B')
        
        if not movimentos:
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
        
        if not movimentos:
            return avaliar_tabuleiro(tabuleiro), None
            
        for mov in movimentos:
            proximo_estado = mover_peca(tabuleiro, mov)
            valor, _ = minimax(proximo_estado, profundidade - 1, True)
            if valor < menor_valor:
                menor_valor = valor
                melhor_movimento = mov
        return menor_valor, melhor_movimento

# 5. LOOP PRINCIPAL DO JOGO (Humano vs IA)
if __name__ == "__main__":
    tabuleiro_jogo = inicializar_tabuleiro()
    turno = 'B'  # Você (Brancas) começa
    
    print("========================================")
    print("      DAMAS SIMPLIFICADAS - IA vs VOCÊ  ")
    print("========================================")
    print("➔ Você comanda as BRANCAS ('B') e se move para CIMA.")
    print("➔ A IA comanda as PRETAS ('P') e se move para BAIXO.")
    print("➔ O jogo termina quando um dos lados não tiver movimentos válidos.")
    
    while True:
        imprimir_tabuleiro(tabuleiro_jogo)
        placar = avaliar_tabuleiro(tabuleiro_jogo)
        print(f"Saldo de Peças (Brancas - Pretas): {placar}")
        
        # --- TURNO DO JOGADOR HUMANO ---
        if turno == 'B':
            print("\n--- SEU TURNO (Brancas 'B') ---")
            movimentos_possiveis = obter_movimentos_validos(tabuleiro_jogo, 'B')
            
            if not movimentos_possiveis:
                print("\n Dominação Completa! Você não tem mais movimentos. A IA VENCEU!")
                break
                
            # Listar opções numeradas para o usuário
            print("Escolha uma das jogadas válidas abaixo:")
            for indice, mov in enumerate(movimentos_possiveis):
                print(f"[{indice}] Mover de ({mov[0]}, {mov[1]}) para ({mov[2]}, {mov[3]})")
            
            # Validar entrada do usuário
            while True:
                try:
                    escolha = int(input("\nDigite o número da sua jogada: "))
                    if 0 <= escolha < len(movimentos_possiveis):
                        jogada_escolhida = movimentos_possiveis[escolha]
                        break
                    print("Número fora da lista! Escolha uma opção válida.")
                except ValueError:
                    print("Entrada inválida! Digite apenas o número correspondente à jogada.")
            
            # Aplicar movimento do humano e passar o turno
            tabuleiro_jogo = mover_peca(tabuleiro_jogo, jogada_escolhida)
            turno = 'P'
            
        # --- TURNO DA INTELIGÊNCIA ARTIFICIAL ---
        else:
            print("\n--- TURNO DA IA (Pretas 'P') ---")
            movimentos_possiveis = obter_movimentos_validos(tabuleiro_jogo, 'P')
            
            if not movimentos_possiveis:
                print("\n Parabéns! A IA ficou sem movimentos. VOCÊ VENCEU!")
                break
                
            print("[IA Pensando...] Analisando cenários futuros com MiniMax...")
            
            # Como a IA é as Pretas ('P'), ela quer MINIMIZAR o valor da função (maximizando=False)
            # Profundidade 4 oferece um bom desafio sem travar o terminal
            _, melhor_jogada_ia = minimax(tabuleiro_jogo, profundidade=4, maximizando=False)
            
            if melhor_jogada_ia:
                ri, ci, rf, cf = melhor_jogada_ia
                print(f"➔ A IA escolheu: mover de ({ri}, {ci}) para ({rf}, {cf})")
                tabuleiro_jogo = mover_peca(tabuleiro_jogo, melhor_jogada_ia)
            else:
                # Fallback de segurança caso a árvore retorne nula
                tabuleiro_jogo = mover_peca(tabuleiro_jogo, movimentos_possiveis[0])
                
            turno = 'B'