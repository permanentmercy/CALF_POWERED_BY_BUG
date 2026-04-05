import torch

from sklearn.decomposition import PCA

from transformers.models.gpt2.modeling_gpt2 import GPT2Model


def load_gpt2_model_local(**kwargs):
    """优先从本地缓存加载GPT2模型，如果没有本地模型才从huggingface下载"""
    try:
        # 首先尝试仅从本地缓存加载（如果之前下载过，就不需要连接huggingface）
        print("正在尝试从本地缓存加载GPT2模型...")
        return GPT2Model.from_pretrained('gpt2', local_files_only=True, **kwargs)
    except (OSError, FileNotFoundError, ConnectionError):
        # 如果本地没有模型文件，才从huggingface下载
        print("本地未找到GPT2模型，准备从huggingface下载...")
        return GPT2Model.from_pretrained('gpt2', **kwargs)


model = load_gpt2_model_local(output_attentions=True, output_hidden_states=True)

wte = model.wte.state_dict()['weight'].cpu().numpy()

pca = PCA(n_components=500)

wte_pca = pca.fit_transform(wte.T)

torch.save(wte_pca, "wte_pca_500.pt")