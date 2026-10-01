import tkinter as tk
from tkinter import messagebox
import copy

# =====================================================================
# 1. LÓGICA DO JOGO DA VALIDAÇÃO (MINIMAX)
# =====================================================================

def inicializar_tabuleiro():
    tabuleiro = [[' ' for _ in range(8)] for _ in range(8)]
    for r in range(3):
        for c in range(8):
            if (r + c) % 2 == 1:
                tabuleiro[r][c] = 'P'  # Pretas (IA)
    for r in range(5, 8):
        for c in range(8):
            if (r + c) % 2 == 1:
                tabuleiro[r][c] = 'B'  # Brancas (Humano)
    return tabuleiro

def avaliar_tabuleiro(tabuleiro):
    brancas = sum(linha.count('B') for linha in tabuleiro)
    pretas = sum(linha.count('P') for linha in tabuleiro)
    return brancas - pretas

def obter_movimentos_validos(tabuleiro, jogador):
    movimentos = []
    direcao = -1 if jogador == 'B' else 1
    for r in range(8):
        for c in range(8):
            if tabuleiro[r][c] == jogador:
                for dc in [-1, 1]:
                    nr, nc = r + direcao, c + dc
                    if 0 <= nr < 8 and 0 <= nc < 8 and tabuleiro[nr][nc] == ' ':
                        movimentos.append((r, c, nr, nc))
    return movimentos

def mover_peca(tabuleiro, movimento):
    tab_copia = copy.deepcopy(tabuleiro)
    ri, ci, rf, cf = movimento
    tab_copia[rf][cf] = tab_copia[ri][ci]
    tab_copia[ri][ci] = ' '
    return tab_copia

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

# =====================================================================
# 2. INTERFACE GRÁFICA (TKINTER)
# =====================================================================

class DamasGrafico:
    def __init__(self, janela):
        self.janela = janela
        self.janela.title("Damas Simplificadas - IA (MiniMax)")
        
        self.tabuleiro = inicializar_tabuleiro()
        self.turno = 'B'  # Humano começa
        self.peca_selecionada = None
        self.movimentos_destaque = []
        
        self.botoes = [[None for _ in range(8)] for _ in range(8)]
        self.criar_malha_botoes()
        self.atualizar_interface()

    def criar_malha_botoes(self):
        # Cria a grelha visual de 8x8 botões
        for r in range(8):
            for c in range(8):
                # Define o padrão xadrez de cores de fundo estáticas
                cor_fundo = "#DDBB99" if (r + c) % 2 == 0 else "#B58863"
                
                botao = tk.Button(
                    self.janela, 
                    font=('Helvetica', 24, 'bold'),
                    width=4, height=2,
                    bg=cor_fundo,
                    command=lambda row=r, col=c: self.clique_quadrado(row, col)
                )
                botao.grid(row=r, column=c)
                self.botoes[r][c] = botao

    def atualizar_interface(self):
        # Atualiza o texto de cada botão com base na matriz lógica do jogo
        for r in range(8):
            for c in range(8):
                conteudo = self.tabuleiro[r][c]
                if conteudo == 'B':
                    self.botoes[r][c].config(text="⚪", fg="white")
                elif conteudo == 'P':
                    self.botoes[r][c].config(text="⚫", fg="black")
                else:
                    self.botoes[r][c].config(text="")
                
                # Restaura as cores padrão do tabuleiro
                cor_padrao = "#DDBB99" if (r + c) % 2 == 0 else "#B58863"
                self.botoes[r][c].config(bg=cor_padrao)

        # Destaca visualmente as jogadas possíveis se houver uma peça selecionada
        for (_, _, rf, cf) in self.movimentos_destaque:
            self.botoes[rf][cf].config(bg="#99FF99") # Verde claro para destinos válidos

    def clique_quadrado(self, r, c):
        if self.turno != 'B':
            return # Bloqueia cliques se for a vez da IA

        # Cenário A: O utilizador clica num destino realçado a verde para mover
        for (ri, ci, rf, cf) in self.movimentos_destaque:
            if (r, c) == (rf, cf):
                self.tabuleiro = mover_peca(self.tabuleiro, (ri, ci, rf, cf))
                self.peca_selecionada = None
                self.movimentos_destaque = []
                self.atualizar_interface()
                
                # Verifica se o Humano venceu
                if not obter_movimentos_validos(self.tabuleiro, 'P'):
                    messagebox.showinfo("Fim de Jogo", "Parabéns! Venceu a IA!")
                    self.janela.quit()
                    return
                
                self.turno = 'P'
                # Executa o turno da IA com um pequeno atraso de 400ms para parecer natural
                self.janela.after(400, self.turno_ia)
                return

        # Cenário B: O utilizador clica numa peça sua para ver as jogadas disponíveis
        if self.tabuleiro[r][c] == 'B':
            self.peca_selecionada = (r, c)
            todos_movimentos = obter_movimentos_validos(self.tabuleiro, 'B')
            # Filtra apenas os movimentos que partem da peça clicada
            self.movimentos_destaque = [m for m in todos_movimentos if m[0] == r and m[1] == c]
            self.atualizar_interface()
        else:
            # Clicar numa casa vazia ou inimiga limpa a seleção
            self.peca_selecionada = None
            self.movimentos_destaque = []
            self.atualizar_interface()

    def turno_ia(self):
        movimentos_ia = obter_movimentos_validos(self.tabuleiro, 'P')
        
        if not movimentos_ia:
            messagebox.showinfo("Fim de Jogo", "A IA não tem jogadas. Vitória do Humano!")
            self.janela.quit()
            return

        # Executa o algoritmo MiniMax com profundidade 4 (rápido e inteligente q.b.)
        # maximizando=False porque a IA joga com as Pretas ('P') e quer diminuir o score
        _, melhor_jogada = minimax(self.tabuleiro, profundidade=4, maximizando=False)

        if melhor_jogada:
            self.tabuleiro = mover_peca(self.tabuleiro, melhor_jogada)
        else:
            # Fallback de segurança
            self.tabuleiro = mover_peca(self.tabuleiro, movimentos_ia[0])

        self.atualizar_interface()

        # Verifica se a IA venceu
        if not obter_movimentos_validos(self.tabuleiro, 'B'):
            messagebox.showinfo("Fim de Jogo", "A IA venceu! Mais sorte na próxima.")
            self.janela.quit()
            return

        self.turno = 'B'

# =====================================================================
# INICIALIZAÇÃO DA APLICAÇÃO
# =====================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = DamasGrafico(root)
    root.mainloop()