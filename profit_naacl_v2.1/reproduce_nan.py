import torch
import torch.nn.functional as F

# Mock attn class from model_9.py
class attn(torch.nn.Module):
    def __init__(self, in_shape, use_attention=True, maxlen=None):
        super(attn, self).__init__()
        self.use_attention = use_attention
        if self.use_attention:
            self.W1 = torch.nn.Linear(in_shape, in_shape)
            self.W2 = torch.nn.Linear(in_shape, in_shape)
            self.V = torch.nn.Linear(in_shape, 1)
        if maxlen != None:
            self.arange = torch.arange(maxlen)

    def forward(self, full, last, lens=None, dim=1):
        if self.use_attention:
            score = self.V(F.tanh(self.W1(last) + self.W2(full)))
            
            if lens is not None:
                mask = self.arange[None, :] < lens[:, None]  # B*30
                score[~mask] = float("-inf")

            attention_weights = F.softmax(score, dim=dim)
            print("Attention Weights:", attention_weights)
            context_vector = attention_weights * full
            context_vector = torch.sum(context_vector, dim=dim)
            return context_vector
        return torch.mean(full, dim=dim)

def test_attn_nan():
    in_shape = 10
    maxlen = 5
    bs = 2
    
    model = attn(in_shape, maxlen=maxlen)
    
    # Create input tensors
    full = torch.randn(bs, maxlen, in_shape)
    last = torch.randn(bs, 1, in_shape)
    
    # Case 1: All valid lengths
    print("Test Case 1: Valid lengths [3, 4]")
    lens = torch.tensor([3, 4])
    output = model(full, last, lens)
    print("Output:", output)
    print("Has NaN:", torch.isnan(output).any().item())
    
    # Case 2: One zero length
    print("\nTest Case 2: Mixed lengths [0, 3]")
    lens = torch.tensor([0, 3])
    output = model(full, last, lens)
    print("Output:", output)
    print("Has NaN:", torch.isnan(output).any().item())

if __name__ == "__main__":
    test_attn_nan()
