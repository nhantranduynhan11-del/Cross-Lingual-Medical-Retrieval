"""Model BGE-M3 "tí hon" dùng chung cho kiểm thử: kiến trúc XLM-R, trọng số ngẫu nhiên (seed cố định),
tokenizer BGE-M3 thật (chỉ tokenizer.json ~17 MB). Không tải trọng số BGE-M3."""
import pytest


def _tokjson():
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download("BAAI/bge-m3", "tokenizer.json")
    except Exception:
        return None


def make_tiny_encoder(path):
    import torch
    from transformers import PreTrainedTokenizerFast, XLMRobertaConfig, XLMRobertaModel
    from mir.encode import M3Encoder
    tok = PreTrainedTokenizerFast(tokenizer_file=path, bos_token="<s>", eos_token="</s>", cls_token="<s>",
                                  sep_token="</s>", pad_token="<pad>", unk_token="<unk>", mask_token="<mask>")
    torch.manual_seed(0)
    cfg = XLMRobertaConfig(vocab_size=len(tok), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
                           intermediate_size=32, max_position_embeddings=520, pad_token_id=tok.pad_token_id)
    return M3Encoder(XLMRobertaModel(cfg), tok, torch.nn.Linear(16, 1), "cpu", False, 512, torch.nn.Linear(16, 16))


@pytest.fixture(scope="session")
def tiny_encoder():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    path = _tokjson()
    if path is None:
        pytest.skip("chưa có tokenizer BGE-M3")
    return make_tiny_encoder(path)
