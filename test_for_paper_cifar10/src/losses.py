import torch
import torch.nn.functional as F

def kd_loss(s, t, *, T, alpha):
    kl = F.kl_div(F.log_softmax(s/T,1), F.softmax(t/T,1), reduction="batchmean")*(T*T)
    return kl

def kd_loss1(s_logits, t_logits, labels, *, T, alpha):
    ce = F.cross_entropy(s_logits, labels)
    kl = F.kl_div(
        F.log_softmax(s_logits / T, 1),
        F.softmax(t_logits / T, 1),
        reduction="batchmean"
    ) * (T*T)
    return alpha*ce + (1-alpha)*kl

def cosine_similarity_loss(o, t, eps=1e-7):
    o = F.normalize(o, dim=1)
    t = F.normalize(t, dim=1)
    so = (torch.mm(o,o.t())+1)/2
    st = (torch.mm(t,t.t())+1)/2
    so /= so.sum(1,keepdim=True)+eps
    st /= st.sum(1,keepdim=True)+eps
    return torch.sum(st * torch.log((st+eps)/(so+eps)))