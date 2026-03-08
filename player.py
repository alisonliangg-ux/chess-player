import chess
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import Optional
from chess_tournament import Player

MODEL_NAME = "your-username/chess-gpt2"

class TransformerPlayer(Player):

    def __init__(self, name: str, model_name: str = MODEL_NAME):
        super().__init__(name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def get_move(self, fen: str) -> Optional[str]:
        board = chess.Board(fen)
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            return None

        move = self._try_generate(fen)

        #if the model outputs something illegal just score all legal moves instead
        if move and self._check_legal(move, board):
            return move

        return self._pick_best(fen, legal_moves)

    def _try_generate(self, fen: str) -> Optional[str]:
        prompt = f"FEN: {fen} MOVE:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        parts = text.split()
        return parts[0] if parts else None

    def _check_legal(self, move_uci: str, board: chess.Board) -> bool:
        try:
            m = chess.Move.from_uci(move_uci)
            return m in board.legal_moves
        except:
            return False

    def _score(self, fen: str, move_uci: str) -> float:
        text = f"FEN: {fen} MOVE: {move_uci}"
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        ids = inputs["input_ids"]
        with torch.no_grad():
            out = self.model(input_ids=ids, labels=ids)
        return -out.loss.item()

    def _pick_best(self, fen: str, legal_moves: list) -> str:
        best = None
        best_score = float("-inf")
        for move in legal_moves:
            uci = move.uci()
            s = self._score(fen, uci)
            if s > best_score:
                best_score = s
                best = uci
        return best
